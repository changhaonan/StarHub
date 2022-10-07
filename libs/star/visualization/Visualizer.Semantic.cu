#include <cmath>
#include <star/common/data_transfer.h>
#include <star/visualization/Visualizer.h>

/* The semantic image drawing method
 */
void star::visualize::DrawSemanticImage(const cv::Mat &semantic_img, const std::map<int, uchar3> &color_dict)
{
	// 1. Transform semantic_img to bgr_img
	auto cols = semantic_img.cols;
	auto rows = semantic_img.rows;

	auto bgr_img = cv::Mat(rows, cols, CV_8UC3);
	for (auto i = 0; i < rows; ++i)
	{
		for (auto j = 0; j < cols; ++j)
		{
			int label = semantic_img.at<int>(i, j);
			auto rgb = color_dict.at(label);
			bgr_img.at<uchar3>(i, j) = make_uchar3(rgb.z, rgb.y, rgb.x);
		}
	}

	// 2. Show image
	cv::imshow("semantic image", bgr_img);
	cv::waitKey(0);
}

void star::visualize::DrawSemanticMap(cudaTextureObject_t semantic_map, const std::map<int, uchar3> &color_dict)
{
	auto semantic_image = downloadSemanticImage(semantic_map);
	DrawSemanticImage(semantic_image, color_dict);
}

void star::visualize::SaveSemanticImage(
	const cv::Mat &semantic_img, const std::map<int, uchar3> &color_dict, const std::string &path)
{
	// 1. Transform semantic_img to bgr_img
	auto cols = semantic_img.cols;
	auto rows = semantic_img.rows;

	auto bgr_img = cv::Mat(rows, cols, CV_8UC3);
	for (auto i = 0; i < rows; ++i)
	{
		for (auto j = 0; j < cols; ++j)
		{
			int label = semantic_img.at<int>(i, j);
			auto rgb = color_dict.at(label);
			bgr_img.at<uchar3>(i, j) = make_uchar3(rgb.z, rgb.y, rgb.x);
		}
	}

	// 2. Save the bgr_img
	SaveRGBImage(bgr_img, path);
}

void star::visualize::SaveSemanticMap(cudaTextureObject_t semantic_map, const std::map<int, uchar3> &color_dict, const std::string &path)
{
	auto semantic_image = downloadSemanticImage(semantic_map);
	SaveSemanticImage(semantic_image, color_dict, path);
}