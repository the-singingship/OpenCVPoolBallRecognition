
// Library for loadinng the images and warp each table to a top-down view
#include "Utilities.h"

// parameters remain the same for every image
const int TABLE_W = 1434;    // table width in mm, columns of image
const int TABLE_H = 683;     // table height in mm,  rows of image

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

    // Warp each table image (index 0 is AllBalls, used for training in later versions)
    for (int file_no = 1; file_no < number_of_images; file_no++)
    {
        if (pool_image[file_no].empty()) continue;

        Mat table = warpTable(pool_image[file_no]);
        imshow(string(pool_files[file_no]) + " warped", table);
        cv::waitKey();
        destroyAllWindows();
    }

    delete[] pool_image;
}