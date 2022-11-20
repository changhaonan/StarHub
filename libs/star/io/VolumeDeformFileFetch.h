#pragma once
#include <string>
#include <boost/filesystem.hpp>
#include "FetchInterface.h"

namespace star
{
	/**
	 * \brief Utility for fetching depth & RGB frames specifically in the format of the VolumeDeform dataset, i.e.
	 * depth frames named as "frame-000000.depth.png" and RBG frames named as "frame-000000.color.png", where zeros are
	 * replaced by the zero-based frame index padded on the left by zeroes to be 6 characters long.
	 */
	class VolumeDeformFileFetch : public FetchInterface
	{
	public:
		using Ptr = std::shared_ptr<VolumeDeformFileFetch>;
		using path = boost::filesystem::path;
		enum class FileType
		{
			color_img_file,
			depth_img_file,
			opticalflow_file,
			point_cloud_file,
			seg_img_file,
		};

		// Just copy the string to data path
		explicit VolumeDeformFileFetch(
			const path &data_path) : m_data_path(data_path) {}
		~VolumeDeformFileFetch() = default;

		// Main interface
		bool FetchDepthImage(size_t cam_idx, size_t frame_idx, cv::Mat &depth_img) override;
		bool FetchDepthImage(size_t cam_idx, size_t frame_idx, void *depth_img) override;
		bool FetchRGBImage(size_t cam_idx, size_t frame_idx, cv::Mat &rgb_img) override;
		bool FetchRGBImage(size_t cam_idx, size_t frame_idx, void *rgb_img) override;
		bool FetchOFImage(size_t cam_idx, size_t frame_idx, cv::Mat &of_img) override;
		bool FetchOFImage(size_t cam_idx, size_t frame_idx, void *of_img) override;
		bool FetchPcd(size_t cam_idx, size_t frame_idx, pcl::PointCloud<pcl::PointXYZRGB>::Ptr pcd) override;
		bool FetchSegImage(size_t cam_idx, size_t frame_idx, cv::Mat &seg_img) override;

	private:
		path m_data_path; // The path prefix for the data
		// A series of naming functions
		path FileNameVolumeDeform(size_t cam_idx, size_t frame_idx, FileType file_type) const;
		path FileNameStar(size_t cam_idx, size_t frame_idx, FileType file_type) const;
	};

}