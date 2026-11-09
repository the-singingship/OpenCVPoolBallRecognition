
#include "Utilities.h"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <iomanip>
#ifdef _WIN32
#include <direct.h>
#define MAKE_FOLDER(p) _mkdir(p)
#else
#include <sys/stat.h>
#define MAKE_FOLDER(p) mkdir(p, 0777)
#endif

// Parameters these are ordinary variables 
const int TABLE_W = 1434;      // table width in mm, so the columns of the image
const int TABLE_H = 683;       // table height in mm, so the rows of the image
int HUE_LOW = 40, HUE_HIGH = 90;   // green cloth, in OpenCV hue units (0-180)
int SAT_LOW = 50;                  // less saturated than this is not cloth
int SAT_OFFSET = 30;               // cloth saturation limit = median cloth saturation of THIS image + this
int EDGE_MARGIN = 50;              // the saturation test is only used this far inside the table (cushions are saturated too)
int VAL_LOW = 60;                  // darker than this is not cloth (dark green balls, shadows)
int KERNEL_SIZE = 7;               // size of the open/close element in pixels
int MIN_BLOB_AREA = 800;           // smaller blobs are noise
int BALL_AREA = 2500;              // fallback only; the real ball area is measured from each image
int MAX_BALLS_PER_BLOB = 16;       // there are only 16 balls, so a blob can never be more than that
int ROUND_MIN_PERCENT = 85;        // area / area of smallest enclosing circle, in %. Single balls are round
int POCKET_RADIUS = 75;            // ignore blobs this close to a pocket
int PATCH_RADIUS = 18;             // pixels around each ball centre used for the features
int HUE_BINS = 12;                 // hue histogram size
int MATCH_DIST = 22;               // evaluation: a detection within a ball radius (43 mm / 2) of a true ball is correct
const int STRICT_MATCH_DIST = 11;  // stricter test: two 43 mm balls overlap with IoU = 0.5 when their centres are 11 mm apart
bool SPLIT_TOUCHING = true;        // split blobs that contain several balls

const char* GT_FILE = "Media/Pool/Ball_Locations_In_Transformed_Image.csv";
const char* OUTPUT_DIR = "Output";     // everything for the report is saved here
bool RUN_SWEEPS = true;                // set to false to skip the sweeps (they take a little while)
int SWEEP_FIRST = 1, SWEEP_LAST = 6;   // which images the sweeps use (1-6 = tables, 0 = AllBalls)

string outputPath(const string& file) { return string(OUTPUT_DIR) + "/" + file; }

// ---------- stage 1 building blocks ----------
Mat warpTable(const Mat& image)
{
    // The brief gives corners as (row, column) but OpenCV wants (x = column, y = row)
    Point2f source[4] = { Point2f(337, 35), Point2f(38, 472), Point2f(1570, 21), Point2f(1888, 451) };  // TL, BL, TR, BR
    Point2f target[4] = { Point2f(0, 0), Point2f(0, TABLE_H - 1), Point2f(TABLE_W - 1, 0), Point2f(TABLE_W - 1, TABLE_H - 1) };
    Mat transform = getPerspectiveTransform(source, target);
    Mat table;
    warpPerspective(image, table, transform, Size(TABLE_W, TABLE_H));
    return table;
}

// Mask where white = not green cloth. clean = false gives the mask before open/close (for the report)
Mat findBallMask(const Mat& table, bool clean = true)
{
    Mat hsv, cloth, mask;
    cvtColor(table, hsv, COLOR_BGR2HSV);
    inRange(hsv, Scalar(HUE_LOW, SAT_LOW, VAL_LOW), Scalar(HUE_HIGH, 255, 255), cloth);
    // Dark green balls (6, 14) have the same hue as the cloth but are more saturated.
    // Inside the playing area, anything too saturated is NOT cloth. Near the cushions we skip this test.
    Mat inner = Mat::zeros(cloth.size(), CV_8UC1);
    inner(Rect(EDGE_MARGIN, EDGE_MARGIN, cloth.cols - 2 * EDGE_MARGIN, cloth.rows - 2 * EDGE_MARGIN)) = 255;
    vector<Mat> ch;
    split(hsv, ch);                                   // ch[1] = saturation
    vector<uchar> sats;                               // saturation of the cloth pixels in the middle of the table
    for (int r = 0; r < cloth.rows; r += 2)
        for (int c = 0; c < cloth.cols; c += 2)
            if (cloth.at<uchar>(r, c) && inner.at<uchar>(r, c)) sats.push_back(ch[1].at<uchar>(r, c));
    if (!sats.empty())
    {
        nth_element(sats.begin(), sats.begin() + sats.size() / 2, sats.end());
        int cap = sats[sats.size() / 2] + SAT_OFFSET;       // learned from this image, not a fixed number
        Mat too_saturated;
        inRange(hsv, Scalar(0, cap + 1, 0), Scalar(179, 255, 255), too_saturated);
        bitwise_and(too_saturated, inner, too_saturated);
        cloth = cloth - too_saturated;                      // those pixels are not cloth
    }
    mask = 255 - cloth;
    if (clean)
    {
        Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(KERNEL_SIZE, KERNEL_SIZE));
        morphologyEx(mask, mask, MORPH_OPEN, kernel);
        morphologyEx(mask, mask, MORPH_CLOSE, kernel);
    }
    return mask;
}

bool nearPocket(Point2f p)
{
    Point2f pockets[6] = { Point2f(0, 0), Point2f(TABLE_W / 2, 0), Point2f(TABLE_W - 1, 0),
                           Point2f(0, TABLE_H - 1), Point2f(TABLE_W / 2, TABLE_H - 1), Point2f(TABLE_W - 1, TABLE_H - 1) };
    for (int i = 0; i < 6; i++)
        if (norm(p - pockets[i]) < POCKET_RADIUS) return true;
    return false;
}

// One centre (column, row) per ball. If split_touching is true, big blobs are split with k-means
// How round is a blob? 1.0 = perfect circle. Two touching balls (peanut) score lower.
double roundness(const vector<Point>& contour)
{
    Point2f c; float r;
    minEnclosingCircle(contour, c, r);
    return contourArea(contour) / (CV_PI * r * r);
}

vector<Point2f> detectBalls(const Mat& mask, bool split_touching)
{
    vector<vector<Point>> contours;
    findContours(mask.clone(), contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    // Step 1: measure the one-ball area from this image = median area of the blobs that look like single balls
    vector<double> single_areas;
    for (size_t i = 0; i < contours.size(); i++)
    {
        double area = contourArea(contours[i]);
        if (area > 0.6 * BALL_AREA && area < 1.4 * BALL_AREA && roundness(contours[i]) * 100 >= ROUND_MIN_PERCENT)
            single_areas.push_back(area);
    }
    double ball_area = BALL_AREA;
    if (single_areas.size() >= 3)
    {
        sort(single_areas.begin(), single_areas.end());
        ball_area = single_areas[single_areas.size() / 2];
    }

    vector<Point2f> centres;
    for (size_t i = 0; i < contours.size(); i++)
    {
        double area = contourArea(contours[i]);
        if (area < MIN_BLOB_AREA) continue;

        // Step 2: round blobs that are not too big = one ball. Otherwise divide by the ball area, with a cap
        int count = 1;
        if (split_touching)
        {
            bool round_enough = roundness(contours[i]) * 100 >= ROUND_MIN_PERCENT;
            if (!(round_enough && area < 1.7 * ball_area))
                count = (int)floor(area / ball_area + 0.5);
        }
        if (count < 1) count = 1;
        if (count > MAX_BALLS_PER_BLOB) count = MAX_BALLS_PER_BLOB;

        if (count == 1)
        {
            Moments m = moments(contours[i]);
            Point2f centre((float)(m.m10 / m.m00), (float)(m.m01 / m.m00));
            if (!nearPocket(centre)) centres.push_back(centre);
        }
        else
        {
            Mat blob = Mat::zeros(mask.size(), CV_8UC1);
            drawContours(blob, contours, (int)i, Scalar(255), FILLED);
            vector<Point> pixels;
            findNonZero(blob, pixels);

            Mat data((int)pixels.size(), 2, CV_32F);
            for (size_t p = 0; p < pixels.size(); p++)
            {
                data.at<float>((int)p, 0) = (float)pixels[p].x;
                data.at<float>((int)p, 1) = (float)pixels[p].y;
            }
            Mat labels, cluster_centres;
            kmeans(data, count, labels, TermCriteria(TermCriteria::EPS + TermCriteria::MAX_ITER, 20, 1.0),
                3, KMEANS_PP_CENTERS, cluster_centres);

            for (int k = 0; k < count; k++)
            {
                Point2f centre(cluster_centres.at<float>(k, 0), cluster_centres.at<float>(k, 1));
                if (!nearPocket(centre)) centres.push_back(centre);
            }
        }
    }
    return centres;
}

// ---------- ground truth ----------
struct GroundTruth
{
    Point2f position[16];   // (column, row); index 0-14 = balls 1-15, index 15 = white
    bool present[16];
};

bool loadGroundTruth(const string& image_name, GroundTruth& gt)
{
    ifstream file(GT_FILE);
    if (!file.is_open()) { cout << "Could not open " << GT_FILE << endl; return false; }

    string line;
    while (getline(file, line))
    {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        vector<string> cells;
        stringstream ss(line);
        string cell;
        while (getline(ss, cell, ',')) cells.push_back(cell);
        if (cells.empty() || cells[0] != image_name) continue;

        for (int b = 0; b < 16; b++)
        {
            size_t row_cell = 1 + 2 * b, col_cell = 2 + 2 * b;
            gt.present[b] = false;
            if (col_cell < cells.size() && !cells[row_cell].empty() && !cells[col_cell].empty())
            {
                gt.position[b] = Point2f(stof(cells[col_cell]), stof(cells[row_cell]));
                gt.present[b] = true;
            }
        }
        return true;
    }
    return false;
}

vector<int> matchBalls(const vector<Point2f>& detections, const GroundTruth& gt)
{
    vector<int> match(16, -1);
    vector<bool> used(detections.size(), false);
    for (int b = 0; b < 16; b++)
    {
        if (!gt.present[b]) continue;
        int best = -1;
        float best_dist = (float)MATCH_DIST;
        for (size_t i = 0; i < detections.size(); i++)
        {
            float d = (float)norm(detections[i] - gt.position[b]);
            if (!used[i] && d < best_dist) { best_dist = d; best = (int)i; }
        }
        if (best >= 0) { used[best] = true; match[b] = best; }
    }
    return match;
}

string labelName(int b)
{
    if (b < 0) return "?";
    if (b == 15) return "W";
    return to_string(b + 1);
}

// ---------- stage 2 building blocks ----------
vector<float> ballFeatures(const Mat& hsv_image, Point2f centre)
{
    vector<float> f(HUE_BINS + 4, 0.0f);
    int pixel_count = 0, white = 0, dark = 0, coloured = 0;
    float sum_s = 0, sum_v = 0;

    for (int dy = -PATCH_RADIUS; dy <= PATCH_RADIUS; dy++)
    {
        for (int dx = -PATCH_RADIUS; dx <= PATCH_RADIUS; dx++)
        {
            if (dx * dx + dy * dy > PATCH_RADIUS * PATCH_RADIUS) continue;
            int x = (int)centre.x + dx, y = (int)centre.y + dy;
            if (x < 0 || y < 0 || x >= hsv_image.cols || y >= hsv_image.rows) continue;

            Vec3b p = hsv_image.at<Vec3b>(y, x);
            int h = p[0], s = p[1], v = p[2];
            pixel_count++;
            if (v < 70) dark++;
            else if (s < 60 && v > 150) white++;
            else
            {
                coloured++;
                sum_s += s;
                sum_v += v;
                int bin = h * HUE_BINS / 180;
                if (bin >= HUE_BINS) bin = HUE_BINS - 1;
                f[2 + bin] += 1;
            }
        }
    }
    if (pixel_count == 0) return f;

    f[0] = white / (float)pixel_count;
    f[1] = dark / (float)pixel_count;
    for (int b = 0; b < HUE_BINS; b++) f[2 + b] /= pixel_count;
    if (coloured > 0)
    {
        f[2 + HUE_BINS] = sum_s / coloured / 255.0f;
        f[3 + HUE_BINS] = sum_v / coloured / 255.0f;
    }
    return f;
}

float featureDistance(const vector<float>& a, const vector<float>& b)
{
    float sum = 0;
    for (size_t i = 0; i < a.size(); i++) sum += (a[i] - b[i]) * (a[i] - b[i]);
    return sqrt(sum);
}

vector<vector<float>> trainReferences(const Mat& all_balls_image, const string& name)
{
    vector<vector<float>> reference(16);
    GroundTruth gt;
    if (all_balls_image.empty() || !loadGroundTruth(name, gt)) return reference;

    Mat table = warpTable(all_balls_image);
    Mat hsv;
    cvtColor(table, hsv, COLOR_BGR2HSV);
    for (int b = 0; b < 16; b++)
        if (gt.present[b]) reference[b] = ballFeatures(hsv, gt.position[b]);
    return reference;
}

int classifyBall(const vector<float>& features, const vector<vector<float>>& reference)
{
    int best = -1;
    float best_dist = 1e9f;
    for (int b = 0; b < 16; b++)
    {
        if (reference[b].empty()) continue;
        float d = featureDistance(features, reference[b]);
        if (d < best_dist) { best_dist = d; best = b; }
    }
    return best;
}

struct Candidate { float dist; int ball; int label; };
bool byDistance(const Candidate& a, const Candidate& b) { return a.dist < b.dist; }

vector<int> assignLabels(const vector<vector<float>>& features, const vector<vector<float>>& reference)
{
    vector<Candidate> all;
    for (size_t i = 0; i < features.size(); i++)
        for (int l = 0; l < 16; l++)
            if (!reference[l].empty())
            {
                Candidate c = { featureDistance(features[i], reference[l]), (int)i, l };
                all.push_back(c);
            }
    sort(all.begin(), all.end(), byDistance);

    vector<int> label(features.size(), -1);
    vector<bool> label_used(16, false);
    for (size_t k = 0; k < all.size(); k++)
    {
        if (label[all[k].ball] < 0 && !label_used[all[k].label])
        {
            label[all[k].ball] = all[k].label;
            label_used[all[k].label] = true;
        }
    }
    return label;
}

// ---------- confusion matrix ----------
void addToConfusion(const vector<int>& match, const vector<int>& predicted, Mat& confusion)
{
    for (int b = 0; b < 16; b++)
    {
        if (match[b] < 0) continue;
        int p = predicted[match[b]];
        if (p < 0) p = 16;
        confusion.at<int>(b, p)++;
    }
}

int confusionCorrect(const Mat& confusion)
{
    int correct = 0;
    for (int r = 0; r < 16; r++) correct += confusion.at<int>(r, r);
    return correct;
}

double confusionAccuracy(const Mat& confusion)
{
    int total = 0;
    for (int r = 0; r < 16; r++)
        for (int c = 0; c < 17; c++) total += confusion.at<int>(r, c);
    return total > 0 ? confusionCorrect(confusion) / (double)total : 0.0;
}

void saveConfusion(const Mat& confusion, const string& filename)
{
    ofstream file(filename);
    file << "true \\ predicted";
    for (int c = 0; c < 17; c++) file << "," << (c == 16 ? "none" : labelName(c));
    file << "\n";
    for (int r = 0; r < 16; r++)
    {
        file << labelName(r);
        for (int c = 0; c < 17; c++) file << "," << confusion.at<int>(r, c);
        file << "\n";
    }
}

// ---------- evidence images for the report ----------
void saveFigures(const string& base, const Mat& original, const Mat& table,
    const vector<Point2f>& balls_no_split, const vector<Point2f>& balls,
    const GroundTruth& gt, const vector<int>& match, const vector<int>& predicted)
{
    string prefix = outputPath(base);
    imwrite(prefix + "_1_original.png", original);
    imwrite(prefix + "_2_warped.png", table);
    imwrite(prefix + "_3a_mask_before_cleaning.png", findBallMask(table, false));
    imwrite(prefix + "_3b_mask_after_cleaning.png", findBallMask(table, true));

    // 4: detections without splitting (touching balls stay merged)
    Mat d1 = table.clone();
    for (size_t i = 0; i < balls_no_split.size(); i++) circle(d1, balls_no_split[i], 25, Scalar(0, 255, 0), 2);
    imwrite(prefix + "_4_detections_no_split.png", d1);

    // which true ball (if any) each detection was matched to
    vector<int> ball_of_detection(balls.size(), -1);
    for (int b = 0; b < 16; b++)
        if (match[b] >= 0) ball_of_detection[match[b]] = b;

    // 5: green = correct detection, yellow = false positive, big red ring = missed ball
    Mat d2 = table.clone();
    for (size_t i = 0; i < balls.size(); i++)
        circle(d2, balls[i], 25, ball_of_detection[i] >= 0 ? Scalar(0, 255, 0) : Scalar(0, 255, 255), 2);
    for (int b = 0; b < 16; b++)
        if (gt.present[b] && match[b] < 0) circle(d2, gt.position[b], 32, Scalar(0, 0, 255), 3);
    imwrite(prefix + "_5_detections_vs_truth.png", d2);

    // 6: labels; green = right label, red = wrong label, yellow = false positive
    Mat d3 = table.clone();
    for (size_t i = 0; i < balls.size(); i++)
    {
        Scalar colour(0, 255, 255);
        if (ball_of_detection[i] >= 0)
            colour = (predicted[i] == ball_of_detection[i]) ? Scalar(0, 255, 0) : Scalar(0, 0, 255);
        circle(d3, balls[i], 25, colour, 2);
        Point text_position((int)balls[i].x - 12, (int)balls[i].y + 8);
        putText(d3, labelName(predicted[i]), text_position, FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 0, 0), 4);
        putText(d3, labelName(predicted[i]), text_position, FONT_HERSHEY_SIMPLEX, 0.8, colour, 2);
    }
    imwrite(prefix + "_6_labelled.png", d3);
}

// ---------- run the whole system on images first..last and measure it ----------
struct Results
{
    int tp, fp, fn;
    double precision, recall, f1, mean_error, acc_nn, acc_unique, end_to_end;
};

Results evaluateAll(Mat* pool_image, char** pool_files, int first, int last, bool save)
{
    Results r = {};
    theRNG().state = 12345;   // k-means starts from random points, so fix the seed: same settings give the same results every time
    vector<vector<float>> reference = trainReferences(pool_image[0], pool_files[0]);
    Mat confusion_nn = Mat::zeros(16, 17, CV_32S);
    Mat confusion_unique = Mat::zeros(16, 17, CV_32S);
    double total_error = 0;

    ofstream per_image;
    if (save)
    {
        per_image.open(outputPath("per_image_results.csv"));
        per_image << "image,truth,found,TP,FP,FN,precision,recall,mean_error_mm\n";
    }

    for (int file_no = first; file_no <= last; file_no++)
    {
        if (pool_image[file_no].empty()) continue;

        Mat table = warpTable(pool_image[file_no]);
        Mat mask = findBallMask(table);
        vector<Point2f> balls = detectBalls(mask, SPLIT_TOUCHING);

        GroundTruth gt;
        if (!loadGroundTruth(pool_files[file_no], gt)) continue;
        vector<int> match = matchBalls(balls, gt);

        int present = 0, tp = 0;
        double error = 0;
        for (int b = 0; b < 16; b++)
        {
            if (!gt.present[b]) continue;
            present++;
            if (match[b] >= 0) { tp++; error += norm(balls[match[b]] - gt.position[b]); }
        }
        int fp = (int)balls.size() - tp, fn = present - tp;
        r.tp += tp; r.fp += fp; r.fn += fn; total_error += error;

        Mat hsv;
        cvtColor(table, hsv, COLOR_BGR2HSV);
        vector<vector<float>> features(balls.size());
        vector<int> predicted_nn(balls.size());
        for (size_t i = 0; i < balls.size(); i++)
        {
            features[i] = ballFeatures(hsv, balls[i]);
            predicted_nn[i] = classifyBall(features[i], reference);
        }
        vector<int> predicted = assignLabels(features, reference);
        addToConfusion(match, predicted_nn, confusion_nn);
        addToConfusion(match, predicted, confusion_unique);

        if (save)
        {
            string name(pool_files[file_no]);
            string base = name.substr(0, name.find('.'));
            saveFigures(base, pool_image[file_no], table, detectBalls(mask, false), balls, gt, match, predicted);
            double p = (tp + fp > 0) ? tp / (double)(tp + fp) : 0.0;
            double rc = (tp + fn > 0) ? tp / (double)(tp + fn) : 0.0;
            per_image << name << "," << present << "," << balls.size() << "," << tp << "," << fp << "," << fn << ","
                << p << "," << rc << "," << (tp > 0 ? error / tp : 0.0) << "\n";
        }
    }

    r.precision = (r.tp + r.fp > 0) ? r.tp / (double)(r.tp + r.fp) : 0.0;
    r.recall = (r.tp + r.fn > 0) ? r.tp / (double)(r.tp + r.fn) : 0.0;
    r.f1 = (r.precision + r.recall > 0) ? 2 * r.precision * r.recall / (r.precision + r.recall) : 0.0;
    r.mean_error = (r.tp > 0) ? total_error / r.tp : 0.0;
    r.acc_nn = confusionAccuracy(confusion_nn);
    r.acc_unique = confusionAccuracy(confusion_unique);
    // fraction of ALL true balls that were both found and given the right number
    r.end_to_end = (r.tp + r.fn > 0) ? confusionCorrect(confusion_unique) / (double)(r.tp + r.fn) : 0.0;

    if (save) saveConfusion(confusion_unique, outputPath("confusion_matrix.csv"));
    return r;
}

void printResults(const string& title, const Results& r)
{
    cout << fixed << setprecision(3);
    cout << title << endl;
    cout << "  Stage 1: TP=" << r.tp << " FP=" << r.fp << " FN=" << r.fn << "  precision=" << r.precision
        << " recall=" << r.recall << " F1=" << r.f1 << " mean error=" << r.mean_error << " mm" << endl;
    cout << "  Stage 2: accuracy nearest neighbour=" << r.acc_nn << ", each ball once=" << r.acc_unique
        << ", end-to-end (found AND right number)=" << r.end_to_end << endl;
}

// ---------- sweep one parameter, keep all the others at their normal value ----------
void runSweep(const string& name, int& parameter, const vector<int>& values,
    Mat* pool_image, char** pool_files)
{
    int original = parameter;
    ofstream file(outputPath("sweep_" + name + ".csv"));
    file << name << ",precision,recall,F1,distance_to_P1R1,mean_error_mm,accuracy_nn,accuracy_unique,end_to_end\n";

    cout << endl << "Sweep of " << name << " (normal value " << original << ")" << endl;
    cout << "value\tprec\trecall\tF1\tdistP1R1\terr_mm\tacc\tend-to-end" << endl;

    vector<double> scores;
    double best = -1;
    int best_value = original;
    double best_pr = 1e9;          // smallest distance to the ideal point P=1, R=1
    int best_pr_value = original;
    for (size_t k = 0; k < values.size(); k++)
    {
        parameter = values[k];
        Results r = evaluateAll(pool_image, pool_files, SWEEP_FIRST, SWEEP_LAST, false);
        scores.push_back(r.end_to_end);
        if (r.end_to_end > best) { best = r.end_to_end; best_value = values[k]; }
        double pr_distance = sqrt((1 - r.precision) * (1 - r.precision) + (1 - r.recall) * (1 - r.recall));
        if (pr_distance < best_pr) { best_pr = pr_distance; best_pr_value = values[k]; }

        cout << values[k] << "\t" << r.precision << "\t" << r.recall << "\t" << r.f1 << "\t" << pr_distance
            << "\t" << r.mean_error << "\t" << r.acc_unique << "\t" << r.end_to_end
            << (values[k] == original ? "   <- normal" : "") << endl;
        file << values[k] << "," << r.precision << "," << r.recall << "," << r.f1 << "," << pr_distance << ","
            << r.mean_error << "," << r.acc_nn << "," << r.acc_unique << "," << r.end_to_end << "\n";
    }
    parameter = original;

    cout << "stage 1 point closest to P=1, R=1 (lecture criterion): " << name << " = " << best_pr_value
        << " (distance " << best_pr << ")" << endl;
    cout << "best end-to-end " << best << " at " << name << " = " << best_value << "; within 0.02 of best at:";
    for (size_t k = 0; k < values.size(); k++)
        if (scores[k] >= best - 0.02) cout << " " << values[k];
    cout << endl;
}

void saveParameters()
{
    ofstream file(outputPath("parameters_used.txt"));
    file << "Table size (mm): " << TABLE_W << " x " << TABLE_H << "\n";
    file << "Cloth hue range: " << HUE_LOW << " to " << HUE_HIGH << " (0-180)\n";
    file << "Minimum saturation for cloth: " << SAT_LOW << "\n";
    file << "Cloth saturation limit = median + " << SAT_OFFSET << "\n";
    file << "Round blob threshold (%): " << ROUND_MIN_PERCENT << ", max balls per blob: " << MAX_BALLS_PER_BLOB << "\n";
    file << "Edge margin for that test: " << EDGE_MARGIN << "\n";
    file << "Minimum brightness for cloth: " << VAL_LOW << "\n";
    file << "Open/close element size: " << KERNEL_SIZE << "\n";
    file << "Minimum blob area: " << MIN_BLOB_AREA << "\n";
    file << "Area of one ball: " << BALL_AREA << "\n";
    file << "Pocket exclusion radius: " << POCKET_RADIUS << "\n";
    file << "Feature patch radius: " << PATCH_RADIUS << "\n";
    file << "Hue histogram bins: " << HUE_BINS << "\n";
    file << "Matching distance (mm): " << MATCH_DIST << " (strict test: " << STRICT_MATCH_DIST << ")\n";
    file << "Splitting of touching balls: " << (SPLIT_TOUCHING ? "on" : "off") << "\n";
}

void MyApplication()
{
    char* file_location = "Media/Pool/";
    char* pool_files[] = {
        "AllBalls.jpeg",
        "Table1.jpeg",
        "Table2.jpeg",
        "Table3.jpeg",
        "Table4.jpeg",
        "Table5.jpeg",
        "Table6.jpeg"
    };

    // Load images
    int number_of_images = sizeof(pool_files) / sizeof(pool_files[0]);
    Mat* pool_image = new Mat[number_of_images];
    //Mat* new_pool_image = new Mat[number_of_images];
    for (int file_no = 0; (file_no < number_of_images); file_no++)
    {
        cout << pool_files[file_no] << endl;
        string filename(file_location);
        filename.append(pool_files[file_no]);
        pool_image[file_no] = imread(filename, -1);

        if (pool_image[file_no].empty())
        {
            cout << "Could not open " << pool_image[file_no] << endl;
        }
        imshow(pool_files[file_no], pool_image[file_no]);

        char pause = cv::waitKey();  // This introduces a delay between frame.  The length of the delay/number can be reduced.
    }
    destroyAllWindows();

    MAKE_FOLDER(OUTPUT_DIR);
    saveParameters();

    // 1. The normal run: saves all the figures and CSV files for the report into the Output folder
    Results normal = evaluateAll(pool_image, pool_files, 1, number_of_images - 1, true);
    cout << endl;
    printResults("NORMAL SETTINGS (tables 1-6)", normal);

    // 1b. Strict matching: a detection only counts if two balls would overlap with IoU >= 0.5
    int normal_match_dist = MATCH_DIST;
    MATCH_DIST = STRICT_MATCH_DIST;
    Results strict = evaluateAll(pool_image, pool_files, 1, number_of_images - 1, false);
    MATCH_DIST = normal_match_dist;
    printResults("STRICT MATCHING (IoU 0.5 equivalent, 11 mm)", strict);

    // 2. Ablation: what does splitting touching balls actually buy us?
    SPLIT_TOUCHING = false;
    Results no_split = evaluateAll(pool_image, pool_files, 1, number_of_images - 1, false);
    SPLIT_TOUCHING = true;
    printResults("WITHOUT splitting touching balls", no_split);

    // 3. Sweeps: change one parameter at a time and watch the results
    if (RUN_SWEEPS)
    {
        runSweep("VAL_LOW", VAL_LOW, { 20, 40, 60, 80, 100, 120 }, pool_image, pool_files);
        runSweep("SAT_LOW", SAT_LOW, { 20, 35, 50, 65, 80, 100 }, pool_image, pool_files);
        runSweep("ROUND_MIN_PERCENT", ROUND_MIN_PERCENT, { 60, 70, 80, 85, 90, 95 }, pool_image, pool_files);
        runSweep("SAT_OFFSET", SAT_OFFSET, { 15, 20, 30, 40, 60, 255 }, pool_image, pool_files);
        runSweep("EDGE_MARGIN", EDGE_MARGIN, { 0, 20, 35, 50, 65, 80 }, pool_image, pool_files);
        runSweep("HUE_LOW", HUE_LOW, { 20, 30, 40, 50, 60 }, pool_image, pool_files);
        runSweep("HUE_HIGH", HUE_HIGH, { 70, 80, 90, 100, 110 }, pool_image, pool_files);
        runSweep("KERNEL_SIZE", KERNEL_SIZE, { 3, 5, 7, 9, 11, 15 }, pool_image, pool_files);
        runSweep("MIN_BLOB_AREA", MIN_BLOB_AREA, { 200, 500, 800, 1100, 1500, 2000 }, pool_image, pool_files);
        runSweep("BALL_AREA", BALL_AREA, { 1500, 2000, 2500, 3000, 3500 }, pool_image, pool_files);
        runSweep("POCKET_RADIUS", POCKET_RADIUS, { 0, 30, 50, 75, 100, 125 }, pool_image, pool_files);
        runSweep("PATCH_RADIUS", PATCH_RADIUS, { 8, 12, 18, 24, 30 }, pool_image, pool_files);
        runSweep("HUE_BINS", HUE_BINS, { 6, 9, 12, 18, 24 }, pool_image, pool_files);
        runSweep("MATCH_DIST", MATCH_DIST, { 5, 11, 15, 22, 30, 40 }, pool_image, pool_files);
    }

    cout << endl << "Done. Figures and CSV files are in the " << OUTPUT_DIR << " folder." << endl;
    delete[] pool_image;
}