
// Library for loadinng the images and warp each table to a top-down view
#include "Utilities.h"

// parameters remain the same for every image
const int TABLE_W = 1434;    // table width in mm, columns of image
const int TABLE_H = 683;     // table height in mm,  rows of image
const int HUE_LOW = 40, HUE_HIGH = 90;   // green cloth, in OpenCV hue units (0-180)
const int SAT_LOW = 50, VAL_LOW = 60;    // paler or darker than this is not cloth (dark green balls, shadows)
const int MIN_BLOB_AREA = 800;           // smaller blobs are noise
const float POCKET_RADIUS = 75;          // ignore blobs this close to a pocket
const int MAX_BLOB_AREA = 3750;          // bigger blobs are several balls touching (fixed in version 5)

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

// One centre (column, row) for every blob that is the right size to be a ball
vector<Point2f> detectBalls(const Mat& mask)
{
    vector<vector<Point>> contours;
    findContours(mask.clone(), contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    vector<Point2f> centres;
    for (size_t i = 0; i < contours.size(); i++)
    {
        double area = contourArea(contours[i]);
        if (area < MIN_BLOB_AREA || area > MAX_BLOB_AREA) continue;

        Moments m = moments(contours[i]);
        Point2f centre((float)(m.m10 / m.m00), (float)(m.m01 / m.m00));
        if (nearPocket(centre)) continue;
        centres.push_back(centre);
    }
    return centres;
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

    for (int file_no = 1; file_no < number_of_images; file_no++)
    {
        if (pool_image[file_no].empty()) continue;

        Mat table = warpTable(pool_image[file_no]);
        Mat mask = findBallMask(table);
        vector<Point2f> balls = detectBalls(mask);
        cout << pool_files[file_no] << ": found " << balls.size() << " balls" << endl;

        Mat display = table.clone();
        for (size_t i = 0; i < balls.size(); i++)
            circle(display, balls[i], 25, Scalar(0, 255, 0), 2);
        imshow(string(pool_files[file_no]) + " result", display);
        imshow(string(pool_files[file_no]) + " mask", mask);
        cv::waitKey();
        destroyAllWindows();
    }

    delete[] pool_image;
}