#pragma once
#include <opencv2/opencv.hpp>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <memory>

namespace star
{

	/**
	 * \brief The virtual class for all the input image fetching.
	 *        The implementation should support threaded fetching.
	 */
	class FetchInterface
	{
	public:
		using Ptr = std::shared_ptr<FetchInterface>;

		// Default contruct and de-construct
		FetchInterface() = default;
		virtual ~FetchInterface() = default;

		// Do not allow copy/assign/move
		FetchInterface(const FetchInterface &) = delete;
		FetchInterface(FetchInterface &&) = delete;
		FetchInterface &operator=(const FetchInterface &) = delete;
		FetchInterface &operator=(FetchInterface &&) = delete;

		// Buffer may be maintained outside fetch object for thread safety
		virtual bool FetchDepthImage(size_t cam_idx, size_t frame_idx, cv::Mat &depth_img) = 0;
		virtual bool FetchDepthImage(size_t cam_dix, size_t frame_idx, void *depth_img) = 0;

		// Should be rgb, in CV_8UC3 format
		virtual bool FetchRGBImage(size_t cam_idx, size_t frame_idx, cv::Mat &rgb_img) = 0;
		virtual bool FetchRGBImage(size_t cam_idx, size_t frame_idx, void *rgb_img) = 0;

		// OpticalFLow, Shoule be CV_32FC2 format
		virtual bool FetchOFImage(size_t cam_idx, size_t frame_idx, cv::Mat &of_img) = 0;
		virtual bool FetchOFImage(size_t cam_idx, size_t frame_idx, void *of_img) = 0;

		// Should be pcl::PointXYZRGB
		virtual bool FetchPcd(size_t cam_idx, size_t frame_idx, pcl::PointCloud<pcl::PointXYZRGB>::Ptr pcd) = 0;

		virtual bool FetchSegImage(size_t cam_idx, size_t frame_idx, cv::Mat &seg_img) = 0;
	};

}