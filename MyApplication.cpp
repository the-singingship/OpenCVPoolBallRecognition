#include "Utilities.h"
#include <fstream>
#include <sstream>

// Parameters are same for all images
const int TABLE_W = 1434;    // table width in mm, columns of image
const int TABLE_H = 683;     // table height in mm, rows of image
const int HUE_LOW = 40, HUE_HIGH = 90;   // green cloth, in OpenCV hue units (0-180)
const int SAT_LOW = 50, VAL_LOW = 60;    // paler or darker than this is not cloth (dark green balls, shadows)
const int MIN_BLOB_AREA = 800;           // smaller blobs are noise
const float POCKET_RADIUS = 75;          // ignore blobs this close to a pocket
const int BALL_AREA = 2500;              // typical area of one ball in the warped image
const char* GT_FILE = "Media/Pool/Ball_Locations_In_Transformed_Image.csv";
const float MATCH_DIST = 30;             // a detection this close to a true ball counts as correct

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

    int total_tp = 0, total_fp = 0, total_fn = 0;
    double total_error = 0;

    for (int file_no = 1; file_no < number_of_images; file_no++)
    {
        if (pool_image[file_no].empty()) continue;

        Mat table = warpTable(pool_image[file_no]);
        Mat mask = findBallMask(table);
        vector<Point2f> balls = detectBalls(mask);

        GroundTruth gt;
        if (!loadGroundTruth(pool_files[file_no], gt)) { cout << "No ground truth for " << pool_files[file_no] << endl; continue; }
        vector<int> match = matchBalls(balls, gt);

        // Stage 1 counts: true positives, false positives, false negatives
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
            << ", TP=" << tp << " FP=" << fp << " FN=" << fn
            << ", mean error " << (tp > 0 ? error / tp : 0.0) << " mm" << endl;

        // green circles = detections, red dots = ground truth
        Mat display = table.clone();
        for (size_t i = 0; i < balls.size(); i++) circle(display, balls[i], 25, Scalar(0, 255, 0), 2);
        for (int b = 0; b < 16; b++) if (gt.present[b]) circle(display, gt.position[b], 5, Scalar(0, 0, 255), FILLED);
        imshow(string(pool_files[file_no]) + " result", display);
        cv::waitKey();
        destroyAllWindows();
    }

    double precision = total_tp / (double)(total_tp + total_fp);
    double recall = total_tp / (double)(total_tp + total_fn);
    double f1 = 2 * precision * recall / (precision + recall);
    cout << "STAGE 1 overall: precision=" << precision << " recall=" << recall << " F1=" << f1
        << " mean error=" << (total_tp > 0 ? total_error / total_tp : 0.0) << " mm" << endl;

    delete[] pool_image;
}