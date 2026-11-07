// Version 7 (complete): version 6 + each ball used once + stage 2 evaluation (accuracy, confusion matrix)
#include "Utilities.h"
#include <fstream>
#include <sstream>
#include <algorithm>

// Parameters are same for all images
const int TABLE_W = 1434;    // table width in mm, columns of image
const int TABLE_H = 683;     // table height in mm, rows of image
const int HUE_LOW = 40, HUE_HIGH = 90;   // green cloth, in OpenCV hue units (0-180)
const int SAT_LOW = 50, VAL_LOW = 60;    // paler or darker than this is not cloth (dark green balls, shadows)
const int MIN_BLOB_AREA = 800;           // smaller blobs are noise
const float POCKET_RADIUS = 75;          // ignore blobs this close to a pocket
const int BALL_AREA = 2500;              // typical area of one ball in the warped image
const char* GT_FILE = "Media/Pool/Ball_Locations_In_Transformed_Image.csv";
const float MATCH_DIST = 22;             // a detection within a ball radius (43 mm / 2) of a true ball counts as correct
const int PATCH_RADIUS = 18;             // we look at this many pixels around each ball centre
const int HUE_BINS = 12;                 // hue histogram size
const int FEATURE_SIZE = HUE_BINS + 4;   // white, dark, hue bins, mean saturation, mean value

// Warp the photo into a top-down view of the table (1 pixel = 1 mm)
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

// Mask where white = something that is not green cloth (balls, pockets, shadows)
Mat findBallMask(const Mat& table)
{
    Mat hsv, cloth, mask;
    cvtColor(table, hsv, COLOR_BGR2HSV);
    inRange(hsv, Scalar(HUE_LOW, SAT_LOW, VAL_LOW), Scalar(HUE_HIGH, 255, 255), cloth);
    mask = 255 - cloth;

    Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(7, 7));
    morphologyEx(mask, mask, MORPH_OPEN, kernel);     // removes small specks
    morphologyEx(mask, mask, MORPH_CLOSE, kernel);    // fills holes made by stripes and numbers
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

// One centre (column, row) per ball. A blob that is too big for one ball is split with k-means
vector<Point2f> detectBalls(const Mat& mask)
{
    vector<vector<Point>> contours;
    findContours(mask.clone(), contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    vector<Point2f> centres;
    for (size_t i = 0; i < contours.size(); i++)
    {
        double area = contourArea(contours[i]);
        if (area < MIN_BLOB_AREA) continue;

        int count = (int)floor(area / BALL_AREA + 0.5);   // how many balls fit in this blob
        if (count < 1) count = 1;

        if (count == 1)
        {
            Moments m = moments(contours[i]);
            Point2f centre((float)(m.m10 / m.m00), (float)(m.m01 / m.m00));
            if (!nearPocket(centre)) centres.push_back(centre);
        }
        else
        {
            // touching balls: cluster the blob's pixels into 'count' groups, one per ball
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

// Ground truth for one image. Index 0-14 = balls 1-15, index 15 = white ball
struct GroundTruth
{
    Point2f position[16];   // (column, row) in the warped table image
    bool present[16];       // false if the ball is not on the table
};

// Reads one image's line from the CSV: name, then (row, column) for ball 1..15 and white
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

// For each true ball find the closest unused detection within MATCH_DIST.
// match[b] = index of the detection for ball b, or -1 if the ball was missed
vector<int> matchBalls(const vector<Point2f>& detections, const GroundTruth& gt)
{
    vector<int> match(16, -1);
    vector<bool> used(detections.size(), false);
    for (int b = 0; b < 16; b++)
    {
        if (!gt.present[b]) continue;
        int best = -1;
        float best_dist = MATCH_DIST;
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

// Describes the patch around a ball centre: how white, how dark, which hues, how saturated and bright
vector<float> ballFeatures(const Mat& hsv_image, Point2f centre)
{
    vector<float> f(FEATURE_SIZE, 0.0f);
    int pixel_count = 0, white = 0, dark = 0, coloured = 0;
    float sum_s = 0, sum_v = 0;

    for (int dy = -PATCH_RADIUS; dy <= PATCH_RADIUS; dy++)
    {
        for (int dx = -PATCH_RADIUS; dx <= PATCH_RADIUS; dx++)
        {
            if (dx * dx + dy * dy > PATCH_RADIUS * PATCH_RADIUS) continue;   // keep it circular
            int x = (int)centre.x + dx, y = (int)centre.y + dy;
            if (x < 0 || y < 0 || x >= hsv_image.cols || y >= hsv_image.rows) continue;

            Vec3b p = hsv_image.at<Vec3b>(y, x);
            int h = p[0], s = p[1], v = p[2];
            pixel_count++;
            if (v < 70) dark++;                       // black ball, shadows
            else if (s < 60 && v > 150) white++;      // white ball, stripes, number circles
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

// Training: AllBalls.jpeg and its ground truth give one reference description per ball
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

// Plain nearest neighbour: the closest reference wins
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

// Nearest neighbour with one extra rule: each numbered ball exists once on the table
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

// Confusion matrix: row = true ball, column = predicted ball (last column = no label given)
void addToConfusion(const vector<int>& match, const vector<int>& predicted, Mat& confusion)
{
    for (int b = 0; b < 16; b++)
    {
        if (match[b] < 0) continue;                   // ball was not even found
        int p = predicted[match[b]];
        if (p < 0) p = 16;
        confusion.at<int>(b, p)++;
    }
}

double confusionAccuracy(const Mat& confusion)
{
    int correct = 0, total = 0;
    for (int r = 0; r < 16; r++)
        for (int c = 0; c < 17; c++)
        {
            total += confusion.at<int>(r, c);
            if (r == c) correct += confusion.at<int>(r, c);
        }
    return total > 0 ? correct / (double)total : 0.0;
}

void printConfusion(const Mat& confusion)
{
    cout << "      ";
    for (int c = 0; c < 17; c++) cout << (c == 16 ? "none" : labelName(c)) << "\t";
    cout << endl;
    for (int r = 0; r < 16; r++)
    {
        cout << "true " << labelName(r) << "\t";
        for (int c = 0; c < 17; c++) cout << confusion.at<int>(r, c) << "\t";
        cout << endl;
    }
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

    // Stage 2 training: one reference description per ball from AllBalls.jpeg
    vector<vector<float>> reference = trainReferences(pool_image[0], pool_files[0]);

    int total_tp = 0, total_fp = 0, total_fn = 0;
    double total_error = 0;
    Mat confusion_nn = Mat::zeros(16, 17, CV_32S);       // plain nearest neighbour
    Mat confusion_unique = Mat::zeros(16, 17, CV_32S);   // nearest neighbour, each ball used once

    for (int file_no = 1; file_no < number_of_images; file_no++)
    {
        if (pool_image[file_no].empty()) continue;

        // ---- stage 1: find the balls ----
        Mat table = warpTable(pool_image[file_no]);
        Mat mask = findBallMask(table);
        vector<Point2f> balls = detectBalls(mask);

        GroundTruth gt;
        if (!loadGroundTruth(pool_files[file_no], gt)) { cout << "No ground truth for " << pool_files[file_no] << endl; continue; }
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
        total_tp += tp; total_fp += fp; total_fn += fn; total_error += error;
        cout << pool_files[file_no] << ": found " << balls.size() << ", truth " << present
            << ", TP=" << tp << " FP=" << fp << " FN=" << fn << endl;

        // ---- stage 2: say which ball each detection is ----
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

        Mat display = table.clone();
        for (size_t i = 0; i < balls.size(); i++)
        {
            circle(display, balls[i], 25, Scalar(0, 255, 0), 2);
            Point text_position((int)balls[i].x - 12, (int)balls[i].y + 8);
            putText(display, labelName(predicted[i]), text_position, FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 0, 0), 4);
            putText(display, labelName(predicted[i]), text_position, FONT_HERSHEY_SIMPLEX, 0.8, Scalar(255, 255, 255), 2);
        }
        for (int b = 0; b < 16; b++)
            if (gt.present[b]) circle(display, gt.position[b], 4, Scalar(0, 0, 255), FILLED);
        imshow(string(pool_files[file_no]) + " result", display);
        cv::waitKey();
        destroyAllWindows();
    }

    // ---- final numbers for the report ----
    double precision = total_tp / (double)(total_tp + total_fp);
    double recall = total_tp / (double)(total_tp + total_fn);
    double f1 = 2 * precision * recall / (precision + recall);
    cout << endl << "STAGE 1 overall: precision=" << precision << " recall=" << recall << " F1=" << f1
        << " mean error=" << (total_tp > 0 ? total_error / total_tp : 0.0) << " mm" << endl;
    cout << "STAGE 2 accuracy (balls found correctly): nearest neighbour=" << confusionAccuracy(confusion_nn)
        << ", with each ball used once=" << confusionAccuracy(confusion_unique) << endl;
    cout << endl << "Confusion matrix (each ball used once):" << endl;
    printConfusion(confusion_unique);

    delete[] pool_image;
}