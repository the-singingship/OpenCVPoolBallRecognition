// Workflow
//   Stage 1: find where every ball is on the table (position in mm on a top-down view)
//   Stage 2: decide which ball each one is (1-15 or white)
//   Then it measures how good both stages are and saves pictures and CSV files for the report.
//
//   photo -> warp to top-down table (1 pixel = 1 mm)
//         -> HSV colour mask: everything that is NOT green cloth is white
//         -> clean the mask (morphology: open then close)
//         -> find blobs (connected white patches) with findContours
//         -> one blob = one ball, or split big blobs into several balls with k-means
//         -> for each ball centre, build a colour feature vector from a small circular patch
//         -> compare with reference vectors learned from AllBalls.jpeg -> ball number
//         -> compare with ground truth CSV -> precision, recall, accuracy, confusion matrix
//
//   ball = the real object.   blob = a connected white patch in the mask (may hold 1 or more balls).

#include "Utilities.h"      // lab framework header: pulls in OpenCV, iostream, using namespace cv/std
#include <fstream>          // ifstream / ofstream: reads the ground truth CSV, writing result files
#include <sstream>          // stringstream: splits CSV line into cells
#include <algorithm>        // sort, nth_element
#include <iomanip>          // setprecision: prints numbers with 3 decimals

// MAKE_FOLDER creates the Output folder. Windows and Linux use different functions for this,
// so we pick the right one depending on the operating system.
#ifdef _WIN32
#include <direct.h>
#define MAKE_FOLDER(p) _mkdir(p)
#else
#include <sys/stat.h>
#define MAKE_FOLDER(p) mkdir(p, 0777)
#endif

// Parameters
const int TABLE_W = 1434;          // table width in mm = number of columns 
const int TABLE_H = 683;           // table height in mm = number of rows

int HUE_LOW = 40, HUE_HIGH = 90;   // green cloth hue range. OpenCV hue is 0-179
int SAT_LOW = 50;                  // pixels less saturated than this (greyish/white) are not cloth
int SAT_OFFSET = 30;               // cloth saturation limit = median cloth saturation.

//   Dark green balls (6, 14) are a bit MORE saturated than the cloth.
int EDGE_MARGIN = 50;              // the saturation test is only used this many pixels inside the table because the cushions are very saturated too and would be false blobs

int VAL_LOW = 60;                  // pixels darker than this (value/brightness) are not cloth (shadows)
int KERNEL_SIZE = 7;               // size of the ellipse used for morphological open/close, in pixels
int MIN_BLOB_AREA = 800;           // blobs smaller than this many pixels are noise, not balls
int BALL_AREA = 2500;              // rough area of one ball in pixels. Only a starting guess: the program

//   measures the real one-ball area from each image (see detectBalls)
int MAX_BALLS_PER_BLOB = 16;       // there are only 16 balls, so one blob can never hold more than that
int ROUND_MIN_PERCENT = 85;        // roundness limit in %: blob area / area of its smallest enclosing circle.

//   A single ball is almost a perfect circle
int POCKET_RADIUS = 75;            // ignore detections closer than this (pixels) to one of the 6 pockets
int PATCH_RADIUS = 18;             // radius (pixels) of the circular patch around a ball centre used for features
int HUE_BINS = 12;                 // number of bins in the hue histogram part of the feature vector
int MATCH_DIST = 22;               // evaluation only: a detection within one ball radius (43 / 2 mm)

//   of a true ball position counts as correct
const int STRICT_MATCH_DIST = 11;  // stricter test: two 43 mm balls overlap with IoU = 0.5 when centres are 11 mm apart
bool SPLIT_TOUCHING = true;        // true = split blobs that contain several balls

const char* GT_FILE = "Media/Pool/Ball_Locations_In_Transformed_Image.csv";  // ground truth (the xlsx saved as CSV)
const char* OUTPUT_DIR = "Output";     // every picture / CSV for the report is saved in this folder
bool RUN_SWEEPS = true;                // set to false to skip the parameter sweeps (they take a while)
int SWEEP_FIRST = 1, SWEEP_LAST = 6;   // which images the sweeps use (1-6 = Table1..6, 0 = AllBalls)

// Small helper: "Output" + "/" + "file.png" = "Output/file.png"
string outputPath(const string& file) { return string(OUTPUT_DIR) + "/" + file; }

// STAGE 1 BUILDING BLOCKS

// warpTable: perspective correction of the image.
// The photo is taken at an angle, so the table looks like a trapezium. We know where the 4
// corners of the table are in the photo and where they should end up (a 1434 x 683 rectangle), so
// getPerspectiveTransform gives us the 3x3 matrix and warpPerspective applies it.
// After this, 1 pixel = 1 mm and distances between balls are real distances.
Mat warpTable(const Mat& image)
{
    // The brief gives the corners as (row, column) but OpenCV wants (x = column, y = row),
    // so every pair below is written the other way round to the brief.
    Point2f source[4] = { Point2f(337, 35), Point2f(38, 472), Point2f(1570, 21), Point2f(1888, 451) };  // TL, BL, TR, BR in the photo
    // Where those corners go in the output image (the corners of a TABLE_W x TABLE_H rectangle)
    Point2f target[4] = { Point2f(0, 0), Point2f(0, TABLE_H - 1), Point2f(TABLE_W - 1, 0), Point2f(TABLE_W - 1, TABLE_H - 1) };
    Mat transform = getPerspectiveTransform(source, target);        // the 3x3 homography matrix
    Mat table;
    warpPerspective(image, table, transform, Size(TABLE_W, TABLE_H)); // apply it to every pixel
    return table;
}

// findBallMask: returns a black/white image where WHITE = "not green cloth" (balls, and noise).
// Steps: convert to HSV -> pick out the cloth colours -> take away the dark-green-ball pixels
//        -> invert -> clean with morphology.
// clean = false returns the mask before cleaning (only used to save a figure for the report).
Mat findBallMask(const Mat& table, bool clean = true)
{
    Mat hsv, cloth, mask;

    // HSV separates colour (hue) from brightness (value), so green is easy to find even if the
    // lighting changes across the table. OpenCV loads images as BGR, so we convert from BGR.
    cvtColor(table, hsv, COLOR_BGR2HSV);

    // inRange gives 255 where ALL three channels are inside the range, 0 elsewhere.
    // So "cloth" is white where the pixel looks like green cloth.
    inRange(hsv, Scalar(HUE_LOW, SAT_LOW, VAL_LOW), Scalar(HUE_HIGH, 255, 255), cloth);

    // solution fix for the dark green balls (6 and 14)
    // Their hue is nearly the same as the cloth, so the rule above thinks they ARE cloth.
    // But they are more saturated than the cloth. Inside the playing area, any pixel that is
    // much more saturated than the typical cloth is therefore NOT cloth.
    // 'inner' is a white rectangle in the middle of the table. We skip a border of EDGE_MARGIN
    // pixels because the cushions are saturated too.
    Mat inner = Mat::zeros(cloth.size(), CV_8UC1);
    inner(Rect(EDGE_MARGIN, EDGE_MARGIN, cloth.cols - 2 * EDGE_MARGIN, cloth.rows - 2 * EDGE_MARGIN)) = 255;

    vector<Mat> ch;
    split(hsv, ch);                                   // split into 3 single-channel images; ch[1] = saturation

    // Collect the saturation of cloth pixels in the middle of the table.
    // We only look at every 2nd row/column (r += 2, c += 2) to keep it fast; plenty of samples anyway.
    vector<uchar> sats;
    for (int r = 0; r < cloth.rows; r += 2)
        for (int c = 0; c < cloth.cols; c += 2)
            if (cloth.at<uchar>(r, c) && inner.at<uchar>(r, c)) sats.push_back(ch[1].at<uchar>(r, c));

    if (!sats.empty())
    {
        // nth_element puts the middle value (the median) at the middle position without sorting everything.
        // The median is not thrown off by the odd ball pixel, unlike the mean.
        nth_element(sats.begin(), sats.begin() + sats.size() / 2, sats.end());
        int cap = sats[sats.size() / 2] + SAT_OFFSET;       // limit is learned from THIS image, not a fixed number

        Mat too_saturated;
        inRange(hsv, Scalar(0, cap + 1, 0), Scalar(179, 255, 255), too_saturated);  // white where saturation > cap
        bitwise_and(too_saturated, inner, too_saturated);   // only count that in the middle of the table
        cloth = cloth - too_saturated;                      // remove those pixels from the cloth mask
    }

    mask = 255 - cloth;     // invert: now white = NOT cloth (balls), black = cloth

    if (clean)
    {
        // Ellipse-shaped element, because balls are round.
        Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(KERNEL_SIZE, KERNEL_SIZE));
        morphologyEx(mask, mask, MORPH_OPEN, kernel);   // OPEN = erode then dilate: removes small white specks
        morphologyEx(mask, mask, MORPH_CLOSE, kernel);  // CLOSE = dilate then erode: fills small holes (numbers, stripes)
    }
    return mask;
}

// nearPocket: true if point p is within POCKET_RADIUS pixels of any of the 6 pockets.
// The dark pocket holes look like "not cloth", so they would show up as fake balls.
// Pockets are at the 4 corners and the middle of the two long sides.
bool nearPocket(Point2f p)
{
    Point2f pockets[6] = { Point2f(0, 0), Point2f(TABLE_W / 2, 0), Point2f(TABLE_W - 1, 0),
                           Point2f(0, TABLE_H - 1), Point2f(TABLE_W / 2, TABLE_H - 1), Point2f(TABLE_W - 1, TABLE_H - 1) };
    for (int i = 0; i < 6; i++)
        if (norm(p - pockets[i]) < POCKET_RADIUS) return true;   // norm of a point difference = distance
    return false;
}

// roundness: how close to a circle is this blob? 1.0 = perfect circle.
// = blob area / area of the smallest circle that contains the whole blob.
// A single ball scores high. Two touching balls (a "peanut") score lower.
double roundness(const vector<Point>& contour)
{
    Point2f c; float r;
    minEnclosingCircle(contour, c, r);                 // smallest circle around the contour (centre c, radius r)
    return contourArea(contour) / (CV_PI * r * r);
}

// detectBalls: from the cleaned mask, return ONE centre (column, row) per ball.
// If split_touching is true, blobs that contain several balls are split with k-means.
vector<Point2f> detectBalls(const Mat& mask, bool split_touching)
{
    // findContours finds the outline of each white blob. It changes the image it is given, so we pass a copy.
    // RETR_EXTERNAL = only outer outlines (ignore holes), CHAIN_APPROX_SIMPLE = store fewer points.
    vector<vector<Point>> contours;
    findContours(mask.clone(), contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    // Step 1: measure the area of ONE ball from this image instead of trusting a fixed number.
    // Blobs that are round and roughly ball-sized are almost certainly single balls,
    // so the median of their areas is a good estimate of the one-ball area.
    vector<double> single_areas;
    for (size_t i = 0; i < contours.size(); i++)
    {
        double area = contourArea(contours[i]);
        if (area > 0.6 * BALL_AREA && area < 1.4 * BALL_AREA && roundness(contours[i]) * 100 >= ROUND_MIN_PERCENT)
            single_areas.push_back(area);
    }
    double ball_area = BALL_AREA;                       // fallback if we did not find enough single balls
    if (single_areas.size() >= 3)
    {
        sort(single_areas.begin(), single_areas.end());
        ball_area = single_areas[single_areas.size() / 2];   // median
    }

    vector<Point2f> centres;                            // the answer: one entry per ball
    for (size_t i = 0; i < contours.size(); i++)
    {
        double area = contourArea(contours[i]);
        if (area < MIN_BLOB_AREA) continue;             // too small: noise

        // Step 2: how many balls are in this blob?
        // A round blob that is not too big is one ball (this also handles a ball with a shadow,
        // or one cut off by the edge, whose area is a bit off). Anything else: divide the area
        // by the one-ball area and round to the nearest whole number.
        int count = 1;
        if (split_touching)
        {
            bool round_enough = roundness(contours[i]) * 100 >= ROUND_MIN_PERCENT;
            if (!(round_enough && area < 1.7 * ball_area))
                count = (int)floor(area / ball_area + 0.5);   // +0.5 then floor = round to nearest
        }
        if (count < 1) count = 1;
        if (count > MAX_BALLS_PER_BLOB) count = MAX_BALLS_PER_BLOB;   // safety cap

        if (count == 1)
        {
            // Centre of the blob = centre of mass, from the image moments (m10/m00, m01/m00).
            Moments m = moments(contours[i]);
            Point2f centre((float)(m.m10 / m.m00), (float)(m.m01 / m.m00));
            if (!nearPocket(centre)) centres.push_back(centre);
        }
        else
        {
            // Several balls touching. Draw the filled blob on a black image, list all its pixels,
            // then k-means groups those pixel coordinates into 'count' clusters.
            // Each cluster centre is one ball centre.
            Mat blob = Mat::zeros(mask.size(), CV_8UC1);
            drawContours(blob, contours, (int)i, Scalar(255), FILLED);
            vector<Point> pixels;
            findNonZero(blob, pixels);                  // coordinates of all white pixels

            // kmeans wants a matrix with one row per sample (here: one row per pixel, columns x and y)
            Mat data((int)pixels.size(), 2, CV_32F);
            for (size_t p = 0; p < pixels.size(); p++)
            {
                data.at<float>((int)p, 0) = (float)pixels[p].x;
                data.at<float>((int)p, 1) = (float)pixels[p].y;
            }
            Mat labels, cluster_centres;
            // stop after 20 iterations or when centres move less than 1 pixel; try 3 random starts and keep the best
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


// GROUND TRUTH (the CSV with the true ball positions, used only for measuring results)
// CSV line format: ImageName, row1, col1, row2, col2, ... for balls 1..15 then white.
// An empty cell means that ball is not on the table in that image.
struct GroundTruth
{
    Point2f position[16];   // (column, row); index 0-14 = balls 1-15, index 15 = white ball
    bool present[16];       // is this ball on the table in this image?
};

// Reads the line for image_name from the CSV into gt. Returns false if the file or the line is missing.
bool loadGroundTruth(const string& image_name, GroundTruth& gt)
{
    ifstream file(GT_FILE);
    if (!file.is_open()) { cout << "Could not open " << GT_FILE << endl; return false; }

    string line;
    while (getline(file, line))
    {
        if (!line.empty() && line.back() == '\r') line.pop_back();   // Windows files end lines with \r\n; drop the \r
        vector<string> cells;
        stringstream ss(line);
        string cell;
        while (getline(ss, cell, ',')) cells.push_back(cell);        // split the line at every comma
        if (cells.empty() || cells[0] != image_name) continue;       // not the line for this image

        for (int b = 0; b < 16; b++)
        {
            size_t row_cell = 1 + 2 * b, col_cell = 2 + 2 * b;       // ball b has its row and column in cells 1+2b and 2+2b
            gt.present[b] = false;
            if (col_cell < cells.size() && !cells[row_cell].empty() && !cells[col_cell].empty())
            {
                // stof = string to float. We store (column, row) = (x, y) to match OpenCV points.
                gt.position[b] = Point2f(stof(cells[col_cell]), stof(cells[row_cell]));
                gt.present[b] = true;
            }
        }
        return true;
    }
    return false;
}

// matchBalls: pair each true ball with the nearest unused detection within MATCH_DIST.
// Returns match[b] = index of the detection matched to ball b, or -1 if the ball was missed.
// "Unused" matters: one detection can only be matched to one ball.
vector<int> matchBalls(const vector<Point2f>& detections, const GroundTruth& gt)
{
    vector<int> match(16, -1);
    vector<bool> used(detections.size(), false);
    for (int b = 0; b < 16; b++)
    {
        if (!gt.present[b]) continue;
        int best = -1;
        float best_dist = (float)MATCH_DIST;            // anything further than this is not a match
        for (size_t i = 0; i < detections.size(); i++)
        {
            float d = (float)norm(detections[i] - gt.position[b]);
            if (!used[i] && d < best_dist) { best_dist = d; best = (int)i; }
        }
        if (best >= 0) { used[best] = true; match[b] = best; }
    }
    return match;
}

// Name of ball b for printing and drawing: "1".."15", or "W" for white, "?" for none
string labelName(int b)
{
    if (b < 0) return "?";
    if (b == 15) return "W";
    return to_string(b + 1);
}


// STAGE 2 BUILDING BLOCKS (recognising which ball is which)
// ballFeatures: describe the colour of a ball as a list of numbers (a "feature vector").
// We look at a circular patch around the ball centre and compute:
//   f[0]            fraction of pixels that are white   (white ball, stripes have white)
//   f[1]            fraction of pixels that are dark    (black 8 ball, numbers)
//   f[2 .. 2+bins]  hue histogram of the coloured pixels, as a fraction of the patch (which colour is it)
//   f[2+bins]       mean saturation of the coloured pixels
//   f[3+bins]       mean value (brightness) of the coloured pixels
// Total length = HUE_BINS + 4 numbers.
vector<float> ballFeatures(const Mat& hsv_image, Point2f centre)
{
    vector<float> f(HUE_BINS + 4, 0.0f);                // all zeros to start
    int pixel_count = 0, white = 0, dark = 0, coloured = 0;
    float sum_s = 0, sum_v = 0;

    // Loop over the square around the centre, but skip pixels outside the circle of radius PATCH_RADIUS
    for (int dy = -PATCH_RADIUS; dy <= PATCH_RADIUS; dy++)
    {
        for (int dx = -PATCH_RADIUS; dx <= PATCH_RADIUS; dx++)
        {
            if (dx * dx + dy * dy > PATCH_RADIUS * PATCH_RADIUS) continue;   // outside the circle
            int x = (int)centre.x + dx, y = (int)centre.y + dy;
            if (x < 0 || y < 0 || x >= hsv_image.cols || y >= hsv_image.rows) continue;  // outside the image

            Vec3b p = hsv_image.at<Vec3b>(y, x);        // at<>(row, column) - note y first
            int h = p[0], s = p[1], v = p[2];
            pixel_count++;
            if (v < 70) dark++;                         // very dark
            else if (s < 60 && v > 150) white++;        // bright and not colourful = white
            else
            {
                coloured++;                             // everything else counts as the ball's colour
                sum_s += s;
                sum_v += v;
                int bin = h * HUE_BINS / 180;           // hue 0-179 -> bin 0..HUE_BINS-1
                if (bin >= HUE_BINS) bin = HUE_BINS - 1;
                f[2 + bin] += 1;
            }
        }
    }
    if (pixel_count == 0) return f;                     // patch was entirely outside the image

    f[0] = white / (float)pixel_count;                  // turn counts into fractions so patch size does not matter
    f[1] = dark / (float)pixel_count;
    for (int b = 0; b < HUE_BINS; b++) f[2 + b] /= pixel_count;
    if (coloured > 0)
    {
        f[2 + HUE_BINS] = sum_s / coloured / 255.0f;    // /255 to scale to 0..1 like the other features
        f[3 + HUE_BINS] = sum_v / coloured / 255.0f;
    }
    return f;
}

// Euclidean distance between two feature vectors (small = similar balls)
float featureDistance(const vector<float>& a, const vector<float>& b)
{
    float sum = 0;
    for (size_t i = 0; i < a.size(); i++) sum += (a[i] - b[i]) * (a[i] - b[i]);
    return sqrt(sum);
}

// trainReferences: "training" = compute the feature vector of each of the 16 balls in
// AllBalls.jpeg (where all balls are on the table and we know where each one is from the
// ground truth CSV). We train ONLY on this image, as the brief requires.
vector<vector<float>> trainReferences(const Mat& all_balls_image, const string& name)
{
    vector<vector<float>> reference(16);                // reference[b] stays empty if ball b is not in the image
    GroundTruth gt;
    if (all_balls_image.empty() || !loadGroundTruth(name, gt)) return reference;

    Mat table = warpTable(all_balls_image);
    Mat hsv;
    cvtColor(table, hsv, COLOR_BGR2HSV);
    for (int b = 0; b < 16; b++)
        if (gt.present[b]) reference[b] = ballFeatures(hsv, gt.position[b]);
    return reference;
}

// Nearest neighbour: the ball whose reference vector is closest to this ball's features
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

// A (detected ball, possible label) pair with the distance between them
struct Candidate { float dist; int ball; int label; };
bool byDistance(const Candidate& a, const Candidate& b) { return a.dist < b.dist; }   // for sort()


// assignLabels: like nearest neighbour, but each label (1-15, white) can only be used ONCE
// per image, because a real table has one of each ball.
// Idea: list every (ball, label) pair, sort by distance, then go through the list from the
// best pair to the worst and accept a pair if both the ball and the label are still free.
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
    sort(all.begin(), all.end(), byDistance);           // closest pairs first

    vector<int> label(features.size(), -1);             // -1 = not assigned yet
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


// CONFUSION MATRIX
// Rows = true ball, columns = predicted ball (last column "none" = no label given).
// Numbers on the diagonal are correct answers.
void addToConfusion(const vector<int>& match, const vector<int>& predicted, Mat& confusion)
{
    for (int b = 0; b < 16; b++)
    {
        if (match[b] < 0) continue;                     // ball was not found, so it cannot be named
        int p = predicted[match[b]];
        if (p < 0) p = 16;                              // no label -> "none" column
        confusion.at<int>(b, p)++;
    }
}

// Sum of the diagonal = number of correctly named balls
int confusionCorrect(const Mat& confusion)
{
    int correct = 0;
    for (int r = 0; r < 16; r++) correct += confusion.at<int>(r, r);
    return correct;
}

// Correctly named / all found balls
double confusionAccuracy(const Mat& confusion)
{
    int total = 0;
    for (int r = 0; r < 16; r++)
        for (int c = 0; c < 17; c++) total += confusion.at<int>(r, c);
    return total > 0 ? confusionCorrect(confusion) / (double)total : 0.0;
}

// Write the matrix to a CSV file (open it in Excel for the report)
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


// EVIDENCE IMAGES FOR THE REPORT (saved into the Output folder, one set per table)
void saveFigures(const string& base, const Mat& original, const Mat& table,
    const vector<Point2f>& balls_no_split, const vector<Point2f>& balls,
    const GroundTruth& gt, const vector<int>& match, const vector<int>& predicted)
{
    string prefix = outputPath(base);                   // e.g. "Output/Table3"
    imwrite(prefix + "_1_original.png", original);      // 1: the photo as given
    imwrite(prefix + "_2_warped.png", table);           // 2: after perspective correction
    imwrite(prefix + "_3a_mask_before_cleaning.png", findBallMask(table, false));  // 3a: raw colour mask
    imwrite(prefix + "_3b_mask_after_cleaning.png", findBallMask(table, true));    // 3b: after open + close

    // 4: detections without splitting (touching balls stay merged as one circle)
    Mat d1 = table.clone();
    for (size_t i = 0; i < balls_no_split.size(); i++) circle(d1, balls_no_split[i], 25, Scalar(0, 255, 0), 2);
    imwrite(prefix + "_4_detections_no_split.png", d1);

    // which true ball (if any) each detection was matched to; -1 = none (a false positive)
    vector<int> ball_of_detection(balls.size(), -1);
    for (int b = 0; b < 16; b++)
        if (match[b] >= 0) ball_of_detection[match[b]] = b;

    // 5: green circle = correct detection, yellow = false positive, big red ring = missed ball.
    // (OpenCV colours are BGR, so Scalar(0,255,0) is green and Scalar(0,0,255) is red.)
    Mat d2 = table.clone();
    for (size_t i = 0; i < balls.size(); i++)
        circle(d2, balls[i], 25, ball_of_detection[i] >= 0 ? Scalar(0, 255, 0) : Scalar(0, 255, 255), 2);
    for (int b = 0; b < 16; b++)
        if (gt.present[b] && match[b] < 0) circle(d2, gt.position[b], 32, Scalar(0, 0, 255), 3);
    imwrite(prefix + "_5_detections_vs_truth.png", d2);

    // 6: predicted label written on each ball; green = right label, red = wrong label, yellow = false positive
    Mat d3 = table.clone();
    for (size_t i = 0; i < balls.size(); i++)
    {
        Scalar colour(0, 255, 255);
        if (ball_of_detection[i] >= 0)
            colour = (predicted[i] == ball_of_detection[i]) ? Scalar(0, 255, 0) : Scalar(0, 0, 255);
        circle(d3, balls[i], 25, colour, 2);
        Point text_position((int)balls[i].x - 12, (int)balls[i].y + 8);
        // text drawn twice: thick black first (outline), then thinner coloured on top, so it is readable on any background
        putText(d3, labelName(predicted[i]), text_position, FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 0, 0), 4);
        putText(d3, labelName(predicted[i]), text_position, FONT_HERSHEY_SIMPLEX, 0.8, colour, 2);
    }
    imwrite(prefix + "_6_labelled.png", d3);
}


// RUN THE WHOLE SYSTEM ON IMAGES first..last AND MEASURE IT
struct Results
{
    int tp, fp, fn;                                     // true positives, false positives, false negatives (stage 1)
    double precision, recall, f1, mean_error, acc_nn, acc_unique, end_to_end;
};

// save = true also writes the figures, per_image_results.csv and confusion_matrix.csv.
// The sweeps call this with save = false because they only need the numbers.
Results evaluateAll(Mat* pool_image, char** pool_files, int first, int last, bool save)
{
    Results r = {};
    // k-means starts from random points, so fix the random seed: same settings -> same results every time
    theRNG().state = 12345;
    vector<vector<float>> reference = trainReferences(pool_image[0], pool_files[0]);   // learn from AllBalls only
    Mat confusion_nn = Mat::zeros(16, 17, CV_32S);      // matrix for plain nearest neighbour
    Mat confusion_unique = Mat::zeros(16, 17, CV_32S);  // matrix for "each label used once"
    double total_error = 0;                             // sum of position errors in mm

    ofstream per_image;
    if (save)
    {
        per_image.open(outputPath("per_image_results.csv"));
        per_image << "image,truth,found,TP,FP,FN,precision,recall,mean_error_mm,"
            "correct_names_nn,correct_names_once,accuracy_nn,accuracy_once,end_to_end\n";
    }

    for (int file_no = first; file_no <= last; file_no++)
    {
        if (pool_image[file_no].empty()) continue;

        // ---- stage 1 ----
        Mat table = warpTable(pool_image[file_no]);
        Mat mask = findBallMask(table);
        vector<Point2f> balls = detectBalls(mask, SPLIT_TOUCHING);

        GroundTruth gt;
        if (!loadGroundTruth(pool_files[file_no], gt)) continue;
        vector<int> match = matchBalls(balls, gt);

        // count: present = balls really on the table, tp = of those found, error = sum of distances (mm)
        int present = 0, tp = 0;
        double error = 0;
        for (int b = 0; b < 16; b++)
        {
            if (!gt.present[b]) continue;
            present++;
            if (match[b] >= 0) { tp++; error += norm(balls[match[b]] - gt.position[b]); }
        }
        int fp = (int)balls.size() - tp;                // detections that matched nothing
        int fn = present - tp;                          // real balls that were not found
        r.tp += tp; r.fp += fp; r.fn += fn; total_error += error;

        // ---- stage 2 ----
        Mat hsv;
        cvtColor(table, hsv, COLOR_BGR2HSV);
        vector<vector<float>> features(balls.size());
        vector<int> predicted_nn(balls.size());
        for (size_t i = 0; i < balls.size(); i++)
        {
            features[i] = ballFeatures(hsv, balls[i]);
            predicted_nn[i] = classifyBall(features[i], reference);   // nearest neighbour answer
        }
        vector<int> predicted = assignLabels(features, reference);    // each-label-once answer
        addToConfusion(match, predicted_nn, confusion_nn);
        addToConfusion(match, predicted, confusion_unique);

        if (save)
        {
            string name(pool_files[file_no]);
            string base = name.substr(0, name.find('.'));             // "Table3.jpeg" -> "Table3"
            saveFigures(base, pool_image[file_no], table, detectBalls(mask, false), balls, gt, match, predicted);

            // how many of the found balls were named correctly in THIS table
            int ok_nn = 0, ok_once = 0;
            for (int b = 0; b < 16; b++)
                if (gt.present[b] && match[b] >= 0)
                {
                    if (predicted_nn[match[b]] == b) ok_nn++;
                    if (predicted[match[b]] == b) ok_once++;
                }
            double p = (tp + fp > 0) ? tp / (double)(tp + fp) : 0.0;     // precision = TP / (TP + FP)
            double rc = (tp + fn > 0) ? tp / (double)(tp + fn) : 0.0;    // recall    = TP / (TP + FN)
            per_image << name << "," << present << "," << balls.size() << "," << tp << "," << fp << "," << fn << ","
                << p << "," << rc << "," << (tp > 0 ? error / tp : 0.0) << ","
                << ok_nn << "," << ok_once << "," << (tp > 0 ? ok_nn / (double)tp : 0.0) << ","
                << (tp > 0 ? ok_once / (double)tp : 0.0) << "," << (present > 0 ? ok_once / (double)present : 0.0) << "\n";
        }
    }

    // ---- totals over all the images (add up TP/FP/FN first, do NOT average the per-image precisions) ----
    r.precision = (r.tp + r.fp > 0) ? r.tp / (double)(r.tp + r.fp) : 0.0;
    r.recall = (r.tp + r.fn > 0) ? r.tp / (double)(r.tp + r.fn) : 0.0;
    r.f1 = (r.precision + r.recall > 0) ? 2 * r.precision * r.recall / (r.precision + r.recall) : 0.0;  // harmonic mean
    r.mean_error = (r.tp > 0) ? total_error / r.tp : 0.0;
    r.acc_nn = confusionAccuracy(confusion_nn);
    r.acc_unique = confusionAccuracy(confusion_unique);
    // end-to-end: share of ALL true balls that were both found and given the right number
    r.end_to_end = (r.tp + r.fn > 0) ? confusionCorrect(confusion_unique) / (double)(r.tp + r.fn) : 0.0;

    if (save) saveConfusion(confusion_unique, outputPath("confusion_matrix.csv"));
    return r;
}

// Print the main numbers to the console
void printResults(const string& title, const Results& r)
{
    cout << fixed << setprecision(3);                   // 3 decimals
    cout << title << endl;
    cout << "  Stage 1: TP=" << r.tp << " FP=" << r.fp << " FN=" << r.fn << "  precision=" << r.precision
        << " recall=" << r.recall << " F1=" << r.f1 << " mean error=" << r.mean_error << " mm" << endl;
    cout << "  Stage 2: accuracy nearest neighbour=" << r.acc_nn << ", each ball once=" << r.acc_unique
        << ", end-to-end (found AND right number)=" << r.end_to_end << endl;
}


// PARAMETER SWEEP
// Change ONE parameter through a list of values, keep all the others at their normal value,
// and record the results. Used in the report to show how sensitive the system is and to
// justify the chosen value.
// 'parameter' is a reference (int&) so we can change the real global variable and put it back.
void runSweep(const string& name, int& parameter, const vector<int>& values,
    Mat* pool_image, char** pool_files)
{
    int original = parameter;                           // remember the normal value so we can restore it
    ofstream file(outputPath("sweep_" + name + ".csv"));
    file << name << ",precision,recall,F1,distance_to_P1R1,mean_error_mm,accuracy_nn,accuracy_unique,end_to_end\n";

    cout << endl << "Sweep of " << name << " (normal value " << original << ")" << endl;
    cout << "value\tprec\trecall\tF1\tdistP1R1\terr_mm\tacc\tend-to-end" << endl;

    vector<double> scores;
    double best = -1;                                   // best end-to-end seen so far
    int best_value = original;
    double best_pr = 1e9;                               // smallest distance to the ideal point P=1, R=1
    int best_pr_value = original;
    for (size_t k = 0; k < values.size(); k++)
    {
        parameter = values[k];                          // try this value
        Results r = evaluateAll(pool_image, pool_files, SWEEP_FIRST, SWEEP_LAST, false);
        scores.push_back(r.end_to_end);
        if (r.end_to_end > best) { best = r.end_to_end; best_value = values[k]; }

        // Lecture criterion: the best setting is the point on the precision/recall plane closest to (1, 1)
        double pr_distance = sqrt((1 - r.precision) * (1 - r.precision) + (1 - r.recall) * (1 - r.recall));
        if (pr_distance < best_pr) { best_pr = pr_distance; best_pr_value = values[k]; }

        cout << values[k] << "\t" << r.precision << "\t" << r.recall << "\t" << r.f1 << "\t" << pr_distance
            << "\t" << r.mean_error << "\t" << r.acc_unique << "\t" << r.end_to_end
            << (values[k] == original ? "   <- normal" : "") << endl;
        file << values[k] << "," << r.precision << "," << r.recall << "," << r.f1 << "," << pr_distance << ","
            << r.mean_error << "," << r.acc_nn << "," << r.acc_unique << "," << r.end_to_end << "\n";
    }
    parameter = original;                               // put the parameter back to its normal value

    cout << "stage 1 point closest to P=1, R=1 (lecture criterion): " << name << " = " << best_pr_value
        << " (distance " << best_pr << ")" << endl;
    cout << "best end-to-end " << best << " at " << name << " = " << best_value << "; within 0.02 of best at:";
    for (size_t k = 0; k < values.size(); k++)
        if (scores[k] >= best - 0.02) cout << " " << values[k];   // every value that is almost as good as the best
    cout << endl;
}

// Save every parameter value to a text file so the report can quote exactly what was used
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


// MAIN ENTRY POINT (called from main.cpp)
void MyApplication()
{
    // ---- load the images (this loading loop is the one from the lab framework, kept as it was) ----
    char* file_location = "Media/Pool/";
    char* pool_files[] = {
        "AllBalls.jpeg",    // index 0: used only for training stage 2
        "Table1.jpeg",
        "Table2.jpeg",
        "Table3.jpeg",
        "Table4.jpeg",
        "Table5.jpeg",
        "Table6.jpeg"
    };

    int number_of_images = sizeof(pool_files) / sizeof(pool_files[0]);   // 7
    Mat* pool_image = new Mat[number_of_images];
    for (int file_no = 0; (file_no < number_of_images); file_no++)
    {
        cout << pool_files[file_no] << endl;
        string filename(file_location);
        filename.append(pool_files[file_no]);
        pool_image[file_no] = imread(filename, -1);     // -1 = load the image exactly as stored

        if (pool_image[file_no].empty())
        {
            cout << "Could not open " << filename << endl;
        }
        imshow(pool_files[file_no], pool_image[file_no]);

        char pause = cv::waitKey();  // wait for a key press before showing the next image
    }
    destroyAllWindows();

    MAKE_FOLDER(OUTPUT_DIR);        // create the Output folder (does nothing if it exists)
    saveParameters();

    // 1. The normal run: saves all the figures and CSV files for the report into the Output folder
    Results normal = evaluateAll(pool_image, pool_files, 1, number_of_images - 1, true);
    cout << endl;
    printResults("NORMAL SETTINGS (tables 1-6)", normal);

    // 1b. Strict matching: a detection only counts if two balls would overlap with IoU >= 0.5 (centres within 11 mm)
    int normal_match_dist = MATCH_DIST;
    MATCH_DIST = STRICT_MATCH_DIST;
    Results strict = evaluateAll(pool_image, pool_files, 1, number_of_images - 1, false);
    MATCH_DIST = normal_match_dist;                     // put it back
    printResults("STRICT MATCHING (IoU 0.5 equivalent, 11 mm)", strict);

    // 2. Ablation: what does splitting touching balls actually buy us? (run with it off, compare)
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
    delete[] pool_image;            // free the memory we allocated with new[]
}