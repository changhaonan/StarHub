#include <cmath>
#include <star/common/data_transfer.h>
#include <star/visualization/Visualizer.h>

/* The depth image drawing methods
 */
void star::visualize::DrawDepthImage(const cv::Mat &depth_img)
{
	double max_depth, min_depth;
	cv::minMaxIdx(depth_img, &min_depth, &max_depth);
	// Visualize depth-image in opencv
	cv::Mat depth_scale;
	cv::convertScaleAbs(depth_img, depth_scale, 255 / max_depth);
	cv::imshow("depth image", depth_scale);
	cv::waitKey(0);
}

void star::visualize::SaveDepthImage(const cv::Mat &depth_img, const std::string &path)
{
	double max_depth, min_depth;
	cv::minMaxIdx(depth_img, &min_depth, &max_depth);
	// Visualize depth-image in opencv
	cv::Mat depth_scale;
	cv::convertScaleAbs(depth_img, depth_scale, 255 / max_depth);
	cv::imwrite(path, depth_scale);
}

void star::visualize::DrawDepthImage(const GArray2D<unsigned short> &depth_img)
{
	const auto depth_cpu = downloadDepthImage(depth_img);
	DrawDepthImage(depth_cpu);
}

void star::visualize::SaveDepthImage(const GArray2D<unsigned short> &depth_img, const std::string &path)
{
	const auto depth_cpu = downloadDepthImage(depth_img);
	SaveDepthImage(depth_cpu, path);
}

void star::visualize::DrawDepthImage(cudaTextureObject_t depth_img)
{
	const auto depth_cpu = downloadDepthImage(depth_img);
	DrawDepthImage(depth_cpu);
}

void star::visualize::SaveDepthImage(cudaTextureObject_t depth_img, const std::string &path)
{
	const auto depth_cpu = downloadDepthImage(depth_img);
	SaveDepthImage(depth_cpu, path);
}

void star::visualize::DrawDepthFloatImage(
	cudaTextureObject_t depth_img)
{
	const auto depth_cpu = downloadDepthFloatImage(depth_img);
	DrawDepthImage(depth_cpu);
}

/* The color image drawing methods
 */
void star::visualize::DrawRGBImage(const cv::Mat &rgb_img)
{
	cv::imshow("color image", rgb_img);
	cv::waitKey(0);
}

void star::visualize::SaveRGBImage(const cv::Mat &rgb_img, const std::string &path)
{
	cv::imwrite(path, rgb_img);
}

void star::visualize::DrawRGBImage(
	const GArray<uchar3> &rgb_img,
	const int rows, const int cols)
{
	const auto rgb_cpu = downloadRGBImage(rgb_img, rows, cols);
	DrawRGBImage(rgb_cpu);
}

void star::visualize::SaveRGBImage(
	const GArray<uchar3> &rgb_img,
	const int rows, const int cols,
	const std::string &path)
{
	const auto rgb_cpu = downloadRGBImage(rgb_img, rows, cols);
	SaveRGBImage(rgb_cpu, path);
}

void star::visualize::DrawNormalizeRGBImage(cudaTextureObject_t rgb_img)
{
	const auto rgb_cpu = downloadNormalizeRGBImage(rgb_img);
	DrawRGBImage(rgb_cpu);
}

void star::visualize::SaveNormalizeRGBImage(cudaTextureObject_t rgb_img, const std::string &path)
{
	const auto rgb_cpu = downloadNormalizeRGBImage(rgb_img);
	cv::Mat rgb_cpu_8uc4;
	rgb_cpu.convertTo(rgb_cpu_8uc4, CV_8UC4, 255);
	SaveRGBImage(rgb_cpu_8uc4, path);
}

void star::visualize::DrawNormalizeRGBDImage(cudaTextureObject_t rgbd_img)
{
	const auto rgbd_cpu = downloadNormalizeRGBImage(rgbd_img); // Download float4 mat
	cv::Mat rgb(rgbd_cpu.rows, rgbd_cpu.cols, CV_32FC3);
	cv::Mat depth(rgbd_cpu.rows, rgbd_cpu.cols, CV_32FC1);

	// forming an array of matrices is a quite efficient operation,
	// because the matrix data is not copied, only the headers
	cv::Mat out[] = {rgb, depth};
	// rgba[0] -> bgr[2], rgba[1] -> bgr[1],
	// rgba[2] -> bgr[0], rgba[3] -> alpha[0]
	int from_to[] = {0, 0, 1, 1, 2, 2, 3, 3};
	cv::mixChannels(&rgbd_cpu, 1, out, 2, from_to, 4);

	// Rescale rgb
	cv::Mat recovered_rgb;
	rgb = 127.5f * (rgb + 1.f);
	rgb.convertTo(recovered_rgb, CV_8UC3);
	DrawRGBImage(recovered_rgb);
	DrawDepthImage(depth);
}

void star::visualize::SaveNormalizeRGBDImage(cudaTextureObject_t rgb_img, const std::string &rgb_path, const std::string &depth_path)
{
	const auto rgbd_cpu = downloadNormalizeRGBImage(rgb_img);
	cv::Mat rgb(rgbd_cpu.rows, rgbd_cpu.cols, CV_32FC3);
	cv::Mat depth(rgbd_cpu.rows, rgbd_cpu.cols, CV_32FC1);

	// forming an array of matrices is a quite efficient operation,
	// because the matrix data is not copied, only the headers
	cv::Mat out[] = {rgb, depth};
	// rgba[0] -> bgr[2], rgba[1] -> bgr[1],
	// rgba[2] -> bgr[0], rgba[3] -> alpha[0]
	int from_to[] = {0, 0, 1, 1, 2, 2, 3, 3};
	cv::mixChannels(&rgbd_cpu, 1, out, 2, from_to, 4);

	// Rescale rgb
	cv::Mat recovered_rgb;
	rgb = 127.5f * (rgb + 1.f);
	rgb.convertTo(recovered_rgb, CV_8UC3);
	SaveRGBImage(recovered_rgb, rgb_path);
	SaveDepthImage(depth, depth_path);
}

void star::visualize::SaveSampledRGBImage(
	cudaTextureObject_t rgb_img,
	GArrayView<unsigned> sampled_indicator,
	const std::string &path)
{
	const auto rgb_cpu = downloadNormalizeRGBImage(rgb_img);
	cv::Mat rgb_cpu_8uc4;
	rgb_cpu.convertTo(rgb_cpu_8uc4, CV_8UC4, 255);

	// Apply sampling mask
	cv::Mat sampled_mask(rgb_cpu_8uc4.rows, rgb_cpu_8uc4.cols, CV_32SC1);
	cudaSafeCall(cudaMemcpy(sampled_mask.data, sampled_indicator.Ptr(), sampled_indicator.Size() * sizeof(unsigned), cudaMemcpyDeviceToHost));
	sampled_mask.convertTo(sampled_mask, CV_8UC1);

	cv::Mat sampled_rgb;
	rgb_cpu_8uc4.copyTo(sampled_rgb, sampled_mask);
	SaveRGBImage(sampled_rgb, path);
}

void star::visualize::DrawColorTimeMap(cudaTextureObject_t color_time_map)
{
	const auto rgb_cpu = rgbImageFromColorTimeMap(color_time_map);
	DrawRGBImage(rgb_cpu);
}

void star::visualize::SaveColorTimeMap(cudaTextureObject_t color_time_map, const std::string &path)
{
	const auto rgb_cpu = rgbImageFromColorTimeMap(color_time_map);
	SaveRGBImage(rgb_cpu, path);
}

void star::visualize::DrawNormalMap(cudaTextureObject_t normal_map)
{
	const auto rgb_cpu = normalMapForVisualize(normal_map);
	DrawRGBImage(rgb_cpu);
}

/* The gray scale image drawing methods
 */
void star::visualize::DrawGrayScaleImage(const cv::Mat &gray_scale_img)
{
	cv::imshow("gray scale image", gray_scale_img);
	cv::waitKey(0);
}

void star::visualize::SaveGrayScaleImage(const cv::Mat &gray_scale_img, const std::string &path)
{
	cv::imwrite(path, gray_scale_img);
}

void star::visualize::DrawGrayScaleImage(cudaTextureObject_t gray_scale_img, float scale)
{
	cv::Mat h_image;
	downloadGrayScaleImage(gray_scale_img, h_image, scale);
	DrawGrayScaleImage(h_image);
}

void star::visualize::SaveGrayScaleImage(cudaTextureObject_t gray_scale_img, const std::string &path, float scale)
{
	cv::Mat h_image;
	downloadGrayScaleImage(gray_scale_img, h_image, scale);
	SaveGrayScaleImage(h_image, path);
}

/* The segmentation mask drawing methods
 */
void star::visualize::MarkSegmentationMask(
	const std::vector<unsigned char> &mask,
	cv::Mat &rgb_img,
	const unsigned sample_rate)
{
	const auto rgb_rows = rgb_img.rows;
	const auto rgb_cols = rgb_img.cols;
	const auto mask_cols = rgb_cols / sample_rate;
	for (auto row = 0; row < rgb_rows; row++)
	{
		for (auto col = 0; col < rgb_cols; col++)
		{
			const auto mask_r = row / sample_rate;
			const auto mask_c = col / sample_rate;
			const auto flatten_idx = mask_c + mask_r * mask_cols;
			const unsigned char mask_value = mask[flatten_idx];
			if (mask_value > 0)
			{
				rgb_img.at<unsigned char>(row, 4 * col + 0) = 255;
				rgb_img.at<unsigned char>(row, 4 * col + 1) = 255;
				// rgb_img.at<unsigned char>(row, 4 * col + 2) = 255;
			}
		}
	}
}

void star::visualize::DrawSegmentMask(
	const std::vector<unsigned char> &mask,
	cv::Mat &rgb_img,
	const unsigned sample_rate)
{
	MarkSegmentationMask(mask, rgb_img, sample_rate);
	DrawRGBImage(rgb_img);
}

void star::visualize::SaveSegmentMask(
	const std::vector<unsigned char> &mask,
	cv::Mat &rgb_img,
	const std::string &path,
	const unsigned sample_rate)
{
	MarkSegmentationMask(mask, rgb_img, sample_rate);
	SaveRGBImage(rgb_img, path);
}

void star::visualize::DrawSegmentMask(
	cudaTextureObject_t mask,
	cudaTextureObject_t normalized_rgb_img,
	const unsigned sample_rate)
{
	// Download the rgb image
	const auto rgb_cpu = downloadNormalizeRGBImage(normalized_rgb_img);
	cv::Mat rgb_cpu_8uc4;
	rgb_cpu.convertTo(rgb_cpu_8uc4, CV_8UC4, 255);

	// Download the segmentation mask
	std::vector<unsigned char> h_mask;
	downloadSegmentationMask(mask, h_mask);

	// Call the drawing functions
	DrawSegmentMask(h_mask, rgb_cpu_8uc4, sample_rate);
}

void star::visualize::SaveSegmentMask(
	cudaTextureObject_t mask,
	cudaTextureObject_t normalized_rgb_img,
	const std::string &path,
	const unsigned sample_rate)
{
	// Download the rgb image
	const auto rgb_cpu = downloadNormalizeRGBImage(normalized_rgb_img);
	cv::Mat rgb_cpu_8uc4;
	rgb_cpu.convertTo(rgb_cpu_8uc4, CV_8UC4, 255);

	// Download the segmentation mask
	std::vector<unsigned char> h_mask;
	downloadSegmentationMask(mask, h_mask);

	// Call the saving methods
	SaveSegmentMask(h_mask, rgb_cpu_8uc4, path, sample_rate);
}

void star::visualize::SaveRawSegmentMask(cudaTextureObject_t mask, const std::string &path)
{
	// Download the segmentation mask
	cv::Mat raw_mask = downloadRawSegmentationMask(mask);

	// Save it to image
	cv::imwrite(path, raw_mask);
}

void star::visualize::DrawRawSegmentMask(cudaTextureObject_t mask)
{
	// Download the segmentation mask
	cv::Mat raw_mask = downloadRawSegmentationMask(mask);

	cv::Mat converted_mask;
	raw_mask.convertTo(converted_mask, CV_8UC1, 255);
	DrawRGBImage(converted_mask);
}

void star::visualize::DrawBinaryMeanfield(cudaTextureObject_t meanfield_q)
{
	cv::Mat h_meanfield_uchar;
	downloadTransferBinaryMeanfield(meanfield_q, h_meanfield_uchar);
	DrawRGBImage(h_meanfield_uchar);
}

void star::visualize::SaveBinaryMeanfield(cudaTextureObject_t meanfield_q, const std::string &path)
{
	cv::Mat h_meanfield_uchar;
	downloadTransferBinaryMeanfield(meanfield_q, h_meanfield_uchar);
	SaveRGBImage(h_meanfield_uchar, path);
}

/* The image pair correspondence method
 */
void star::visualize::DrawImagePairCorrespondence(
	cudaTextureObject_t rgb_0,
	cudaTextureObject_t rgb_1,
	const star::GArray<ushort4> &corr_d)
{
	// Download the data
	std::vector<ushort4> corr;
	corr_d.download(corr);
	cv::Mat normalized_from = downloadNormalizeRGBImage(rgb_0);
	cv::Mat normalized_to = downloadNormalizeRGBImage(rgb_1);
	cv::Mat from, to;
	normalized_from.convertTo(from, CV_8UC4, 255);
	normalized_to.convertTo(to, CV_8UC4, 255);

	// cv programs
	int ind = 0;

	cv::Mat show0 = cv::Mat::zeros(from.rows, to.cols * 2, CV_8UC4);
	cv::Rect rect(0, 0, from.cols, from.rows);
	from.copyTo(show0(rect));
	rect.x = rect.x + from.cols;
	to.copyTo(show0(rect));

	cv::RNG rng(12345);
	srand(time(NULL));
	int step = corr.size() / 10 * 2;
	int goout = 1;
	cv::Mat show;

	cv::namedWindow("Correspondences", cv::WINDOW_AUTOSIZE);
	std::cout << "Only about 100 points are shown, press N to show others, press other keys to exit!" << std::endl;
	while (goout == 1)
	{
		show = show0.clone();
		while (ind < corr.size())
		{
			cv::Point2f a; // = corr[ind].first;
			a.x = corr[ind].x;
			a.y = corr[ind].y;
			cv::Point2f b; // = corr[ind].second;
			b.x = corr[ind].z;
			b.y = corr[ind].w;
			b.x += from.cols;
			cv::Scalar color = cv::Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
			cv::line(show, a, b, color, 1);
			ind = ind + (int)rand() % step;
		}
		imshow("Correspondences", show);
		int c = cv::waitKey();
		switch (c)
		{
		case 'n':
		case 'N':
			goout = 1;
			break;
		default:
			goout = 0;
		}
		ind = 0;
	}
}

void star::visualize::DrawIndicatorMap(
	const GArrayView<unsigned> &indicator,
	size_t img_height,
	size_t img_width)
{
	cv::Mat indicator_map = GetIndicatorMapCV(indicator, img_height, img_width);
	DrawRGBImage(indicator_map);
}

void star::visualize::SaveIndicatorMap(
	const GArrayView<unsigned> &indicator,
	size_t img_height,
	size_t img_width,
	const std::string &img_path)
{
	cv::Mat indicator_map = GetIndicatorMapCV(indicator, img_height, img_width);
	SaveRGBImage(indicator_map, img_path);
}

cv::Mat star::visualize::GetIndicatorMapCV(
	const GArrayView<unsigned> &indicator,
	size_t img_height,
	size_t img_width)
{
	std::vector<unsigned> h_indicator;
	indicator.Download(h_indicator);
	assert(h_indicator.size() == img_height * img_width);

	cv::Mat indicator_map = cv::Mat(img_height, img_width, CV_8UC1);
	for (auto y = 0; y < img_height; y++)
	{
		for (auto x = 0; x < img_width; x++)
		{
			const auto offset = x + y * img_width;
			if (h_indicator[offset] > 0)
			{
				indicator_map.at<unsigned char>(y, x) = 255;
			}
			else
			{
				indicator_map.at<unsigned char>(y, x) = 0;
			}
		}
	}

	return indicator_map;
}

void star::visualize::DrawPixel2DMap(
	const GArrayView<ushort2> &pixel_2d,
	size_t img_height,
	size_t img_width)
{
	cv::Mat pixel_2d_map = GetPixel2DMapCV(pixel_2d, img_height, img_width);
	DrawRGBImage(pixel_2d_map);
}

void star::visualize::SavePixel2DMap(
	const GArrayView<ushort2> &pixel_2d,
	size_t img_height,
	size_t img_width,
	const std::string &img_path)
{
	cv::Mat pixel_2d_map = GetPixel2DMapCV(pixel_2d, img_height, img_width);
	SaveRGBImage(pixel_2d_map, img_path);
}

cv::Mat star::visualize::GetPixel2DMapCV(
	const GArrayView<ushort2> &pixel_2d,
	size_t img_height,
	size_t img_width)
{
	std::vector<ushort2> h_pixel_2d;
	pixel_2d.Download(h_pixel_2d);
	assert(h_pixel_2d.size() <= img_height * img_width); // Smaller

	cv::Mat pixel_2d_map = cv::Mat(img_height, img_width, CV_8UC1);
	pixel_2d_map.setTo(0);

	for (auto i = 0; i < h_pixel_2d.size(); ++i)
	{
		auto x = h_pixel_2d[i].x;
		auto y = h_pixel_2d[i].y;
		pixel_2d_map.at<unsigned char>(y, x) = 255;
	}

	return pixel_2d_map;
}

// Utils for visualize optical-flow
constexpr unsigned RY = 15;
constexpr unsigned YG = 6;
constexpr unsigned GC = 4;
constexpr unsigned CB = 11;
constexpr unsigned BM = 13;
constexpr unsigned MR = 6;
constexpr unsigned ncols = RY + YG + GC + CB + BM + MR;

Eigen::Matrix<float, ncols, 3> gen_color_wheel()
{

	Eigen::Matrix<float, ncols, 3> color_wheel = Eigen::Matrix<float, ncols, 3>::Zero();
	unsigned col = 0;

	// RY
	color_wheel.block(0, 0, RY, 1) = Eigen::Matrix<float, RY, 1>::Ones() * 255.f;
	color_wheel.block(0, 1, RY, 1) = Eigen::Matrix<float, RY, 1>::LinSpaced(RY, 0, RY - 1) * (255.f / float(RY));
	col = col + RY;
	// YG
	color_wheel.block(col, 0, YG, 1) = Eigen::Matrix<float, YG, 1>::Ones() * 255.f - Eigen::Matrix<float, YG, 1>::LinSpaced(YG, 0, YG - 1) * 255.f / float(YG);
	color_wheel.block(col, 1, YG, 1) = Eigen::Matrix<float, YG, 1>::Ones() * 255.f;
	col = col + YG;
	// GC
	color_wheel.block(col, 1, GC, 1) = Eigen::Matrix<float, GC, 1>::Ones() * 255.f;
	color_wheel.block(col, 2, GC, 1) = Eigen::Matrix<float, GC, 1>::LinSpaced(GC, 0, GC - 1) * 255.f / float(GC);
	col = col + GC;
	// CB
	color_wheel.block(col, 1, CB, 1) = Eigen::Matrix<float, CB, 1>::Ones() * 255.f - Eigen::Matrix<float, CB, 1>::LinSpaced(CB, 0, CB - 1) * 255.f / float(CB);
	color_wheel.block(col, 2, CB, 1) = Eigen::Matrix<float, CB, 1>::Ones() * 255.f;
	col = col + CB;
	// BM
	color_wheel.block(col, 2, BM, 1) = Eigen::Matrix<float, BM, 1>::Ones() * 255.f;
	color_wheel.block(col, 0, BM, 1) = Eigen::Matrix<float, BM, 1>::LinSpaced(BM, 0, BM - 1) * 255.f / float(BM);
	col = col + BM;
	// MR
	color_wheel.block(col, 2, MR, 1) = Eigen::Matrix<float, MR, 1>::Ones() * 255.f - Eigen::Matrix<float, MR, 1>::LinSpaced(MR, 0, MR - 1) * 255.f / float(MR);
	color_wheel.block(col, 0, MR, 1) = Eigen::Matrix<float, MR, 1>::Ones() * 255.f;

	return color_wheel;
}

// TODO: fix it to the right color
cv::Mat OpticalFlow2Image(cv::Mat &flow)
{
	cv::Mat flow_uv[2]; // X,Y
	cv::split(flow, flow_uv);
	cv::Mat flow_u = flow_uv[0];
	cv::Mat flow_v = flow_uv[1];
	// Normalize
	cv::Mat flow_u_sq, flow_v_sq, flow_uv_sq;
	cv::multiply(flow_u, flow_u, flow_u_sq);
	cv::multiply(flow_v, flow_v, flow_v_sq);
	cv::add(flow_u_sq, flow_v_sq, flow_uv_sq);

	double min_val;
	double max_val;
	cv::minMaxLoc(flow_uv_sq, &min_val, &max_val);
	float norm = std::sqrt(max_val) + 1e-5;
	flow_u = flow_u / norm;
	flow_v = flow_v / norm;

	// Assign color
	auto img_size = flow_u.size();
	Eigen::Matrix<float, ncols, 3> color_wheel = gen_color_wheel();

	cv::Mat flow_img(cv::Size(img_size.width, img_size.height), CV_8UC3, cv::Scalar(0));
	for (auto i = 0; i < img_size.width; ++i)
	{
		for (auto j = 0; j < img_size.height; ++j)
		{
			float u = flow_u.at<float>(j, i);
			float v = flow_v.at<float>(j, i);
			// Do a truncating
			u = (fabs(u) < 1e-6f) ? 0.f : u;
			v = (fabs(v) < 1e-6f) ? 0.f : v;
			float rad = std::sqrt(u * u + v * v);
			float a = std::atan2(-v, -u) / M_PI;
			float fk = (a + 1.0) / 2.0 * double(ncols - 1);
			unsigned k0 = std::floor(fk);
			unsigned k1 = k0 + 1;
			if (k1 == ncols)
				k1 = 0;
			float f = fk - float(k0);
			for (auto k = 0; k < 3; ++k)
			{ // Go over each color channel
				float col0 = color_wheel(k0, k) / 255.f;
				float col1 = color_wheel(k1, k) / 255.f;
				float col = (1.f - f) * col0 + f * col1;
				if (rad <= 1.f)
				{
					col = 1.f - rad * (1.f - col);
				}
				else
				{
					col = col * 0.75;
				}
				// RGB -> BGR
				flow_img.at<cv::Vec3b>(j, i)[2 - k] = std::floor(col * 255.f);
			}
		}
	}

	return flow_img;
}

void star::visualize::DrawOpticalFlowMap(cudaTextureObject_t opticalflow)
{
	// Download to device memory
	auto opticalflow_mat = downloadOptcalFlowImage(opticalflow);
	cv::Mat img = OpticalFlow2Image(opticalflow_mat);
	cv::imshow("optical flow", img);
	cv::waitKey(0);
}

void star::visualize::SaveOpticalFlowMap(cudaTextureObject_t opticalflow, const std::string &path)
{
	// Download to device memory
	auto opticalflow_mat = downloadOptcalFlowImage(opticalflow);
	cv::Mat img = OpticalFlow2Image(opticalflow_mat);
	cv::imwrite(path, img);
}