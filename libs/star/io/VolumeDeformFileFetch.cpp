#include <opencv2/opencv.hpp>
#include <pcl/io/pcd_io.h>
#include <star/common/OpenCV_CrossPlatform.h>
#include "VolumeDeformFileFetch.h"
// Noise-related
#include <iostream>
#include <star/data_proc/noise_generator.h>


bool star::VolumeDeformFileFetch::FetchDepthImage(size_t cam_idx, size_t frame_idx, cv::Mat& depth_img)
{
	path file_path = FileNameStar(cam_idx, frame_idx, FileType::depth_img_file);
	//Read the image
	cv::Mat raw_depth_img = cv::imread(file_path.string(), CV_ANYCOLOR | CV_ANYDEPTH);
    std::cout << "Depth loaded from " << file_path.string() << " !" << std::endl;
	// Apply noises
//#define APPLY_NOISES
#ifdef APPLY_NOISES
	cv::Mat noise_depth_img = cv::Mat::zeros(depth_img.rows, depth_img.cols, CV_16UC1);
	apply_gaussian_noise(raw_depth_img, noise_depth_img, 0.01f);
	apply_random_corp(noise_depth_img, depth_img, 0.3f);
	
	//apply_gaussian_noise(raw_depth_img, depth_img, 0.03f);
#else
	raw_depth_img.copyTo(depth_img);
#endif // APPLY_NOISES

    return true;
}

bool star::VolumeDeformFileFetch::FetchDepthImage(size_t cam_idx, size_t frame_idx, void *depth_img)
{
    return false;
}

bool star::VolumeDeformFileFetch::FetchRGBImage(size_t cam_idx, size_t frame_idx, cv::Mat& rgb_img)
{
	path file_path = FileNameStar(cam_idx, frame_idx, FileType::color_img_file);
	//Read the image
	rgb_img = cv::imread(file_path.string(), CV_ANYCOLOR | CV_ANYDEPTH);
    std::cout << "RGB loaded from " << file_path.string() << " !" << std::endl;
    return true;
}

bool star::VolumeDeformFileFetch::FetchRGBImage(size_t cam_idx, size_t frame_idx, void* rgb_img)
{
    return false;
}

bool star::VolumeDeformFileFetch::FetchOFImage(size_t cam_idx, size_t frame_idx, cv::Mat& of_img) {
    path file_path = FileNameStar(cam_idx, frame_idx, FileType::opticalflow_file);
    //Read the xml file
    cv::FileStorage fs;
    fs.open(file_path.string(), cv::FileStorage::READ);
    cv::FileNode fn = fs.root();
    for (cv::FileNodeIterator fit = fn.begin(); fit != fn.end(); ++fit)
    {
        cv::FileNode item = *fit;
        std::cout << item.name() << std::endl;
        if (item.name() == "optical_flow") {
            of_img = item.mat();
            return true;
        }
        else if (item.name() == "opticalflow") {
            of_img = item.mat();
            return true;
        }
        else if (item.name() == "OpticalFlow") {
            of_img = item.mat();
            return true;
        }
    }
    return false;
}

bool star::VolumeDeformFileFetch::FetchOFImage(size_t cam_idx, size_t frame_idx, void* of_img) {
    return false;
}

bool star::VolumeDeformFileFetch::FetchPcd(size_t cam_idx, size_t frame_idx, pcl::PointCloud<pcl::PointXYZRGB>::Ptr pcd)
{
    path file_path = FileNameStar(cam_idx, frame_idx, FileType::point_cloud_file);
    pcl::io::loadPCDFile<pcl::PointXYZRGB>(file_path.string(), *pcd);
    std::cout << "PCD loaded from " << file_path.string() << " !" << std::endl;
    return true;
}

boost::filesystem::path star::VolumeDeformFileFetch::FileNameVolumeDeform(size_t cam_idx, size_t frame_idx, FileType file_type) const
{
	//Construct the file_name
	char frame_idx_str[20];
	sprintf(frame_idx_str, "%06d", static_cast<int>(frame_idx));
	char cam_idx_str[20];
	sprintf(cam_idx_str, "%02d", static_cast<int>(cam_idx));
	std::string file_name = "cam-" + std::string(cam_idx_str);
	file_name += "/frame-" + std::string(frame_idx_str);
    switch (file_type)
    {
    case FileType::color_img_file:
        file_name += ".color.png";
        break;
    case FileType::depth_img_file:
        file_name += ".depth.png";
        break;
    case FileType::opticalflow_file:
        file_name += ".of.xml";
        break;
    case FileType::point_cloud_file:
        file_name += ".pcd";
        break;
    default:
        printf("File type is not supported.\n");
        assert(false);
        break;
    }

	//Construct the path
	path file_path = m_data_path / path(file_name);
	return file_path;
}


boost::filesystem::path star::VolumeDeformFileFetch::FileNameStar(size_t cam_idx, size_t frame_idx, FileType file_type) const {
	return FileNameVolumeDeform(cam_idx, frame_idx, file_type);
}