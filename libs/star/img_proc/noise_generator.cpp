#include <star/img_proc/noise_generator.h>

void star::apply_gaussian_noise(
	const cv::Mat &depth_image,
	cv::Mat &noise_depth_image,
	float noise_level)
{
	cv::Mat uniform_noise = cv::Mat::zeros(depth_image.rows, depth_image.cols, CV_16UC1); // unsigned short
	cv::randu(uniform_noise, 0, ushort(noise_level * 1000.f));
	cv::Mat raw_noise_depth_image = depth_image + uniform_noise;
	cv::bitwise_and(raw_noise_depth_image, raw_noise_depth_image, noise_depth_image, (depth_image != 0));
}

void star::apply_random_corp(
	const cv::Mat &depth_image,
	cv::Mat &droped_depth_image,
	float corp_ratio)
{
	unsigned rows = depth_image.rows;
	unsigned cols = depth_image.cols;

	unsigned corp_rows = floor(corp_ratio * rows);
	unsigned corp_cols = floor(corp_ratio * cols);
	unsigned max_rows = rows - corp_rows;
	unsigned max_cols = cols - corp_cols;

	unsigned corp_row_start = std::rand() % max_rows;
	unsigned corp_col_start = std::rand() % max_cols;

	cv::Rect corp_area(corp_col_start, corp_row_start, corp_cols, corp_rows);
	depth_image.copyTo(droped_depth_image);
	cv::Mat roi = droped_depth_image(corp_area); // Set roi and droped
	roi.setTo(0);
}