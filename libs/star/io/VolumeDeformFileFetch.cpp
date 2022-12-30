#include <opencv2/opencv.hpp>
#include <pcl/io/pcd_io.h>
#include <star/common/OpenCV_CrossPlatform.h>
#include "VolumeDeformFileFetch.h"
#include <iostream>
#include <fstream>
#include <nlohmann/json.hpp>
using json = nlohmann::json;

bool star::VolumeDeformFileFetch::FetchDepthImage(size_t cam_idx, size_t frame_idx, cv::Mat &depth_img)
{
    path file_path = FileNameStar(cam_idx, frame_idx, FileType::depth_img_file);
    // Read the image
    cv::Mat raw_depth_img = cv::imread(file_path.string(), CV_ANYCOLOR | CV_ANYDEPTH);
    std::cout << "Depth loaded from " << file_path.string() << " !" << std::endl;
    raw_depth_img.copyTo(depth_img);
    return true;
}

bool star::VolumeDeformFileFetch::FetchDepthImage(size_t cam_idx, size_t frame_idx, void *depth_img)
{
    return false;
}

bool star::VolumeDeformFileFetch::FetchRGBImage(size_t cam_idx, size_t frame_idx, cv::Mat &rgb_img)
{
    path file_path = FileNameStar(cam_idx, frame_idx, FileType::color_img_file);
    // Read the image
    rgb_img = cv::imread(file_path.string(), CV_ANYCOLOR | CV_ANYDEPTH);
    std::cout << "RGB loaded from " << file_path.string() << " !" << std::endl;
    return true;
}

bool star::VolumeDeformFileFetch::FetchRGBImage(size_t cam_idx, size_t frame_idx, void *rgb_img)
{
    return false;
}

bool star::VolumeDeformFileFetch::FetchOFImage(size_t cam_idx, size_t frame_idx, cv::Mat &of_img)
{
    path file_path = FileNameStar(cam_idx, frame_idx, FileType::opticalflow_file);
    // Read the image
    auto of_img_c3 = cv::imread(file_path.string(), cv::IMREAD_UNCHANGED);
    std::cout << "Opricalflow loaded from " << file_path.string() << " !" << std::endl;
    // Take the frist two channels of the mat
    std::vector<cv::Mat> channels(3);
    cv::split(of_img_c3, channels);
    cv::merge(std::vector<cv::Mat>{channels[0], channels[1]}, of_img);
    // Rescale and retype
    of_img.convertTo(of_img, CV_32FC2, 1.0 / 10.0, -1000.0);
    return true;
}

bool star::VolumeDeformFileFetch::FetchOFImage(size_t cam_idx, size_t frame_idx, void *of_img)
{
    return false;
}

bool star::VolumeDeformFileFetch::FetchPcd(size_t cam_idx, size_t frame_idx, pcl::PointCloud<pcl::PointXYZRGB>::Ptr pcd)
{
    path file_path = FileNameStar(cam_idx, frame_idx, FileType::point_cloud_file);
    pcl::io::loadPCDFile<pcl::PointXYZRGB>(file_path.string(), *pcd);
    std::cout << "PCD loaded from " << file_path.string() << " !" << std::endl;
    return true;
}

bool star::VolumeDeformFileFetch::FetchSegImage(size_t cam_idx, size_t frame_idx, cv::Mat &seg_img)
{
    path file_path = FileNameStar(cam_idx, frame_idx, FileType::seg_img_file);
    // Read the image
    seg_img = cv::imread(file_path.string(), cv::IMREAD_UNCHANGED);
    seg_img.convertTo(seg_img, CV_32SC1);
    std::cout << "Segmentation loaded from " << file_path.string() << " !" << std::endl;
    return true;
}

bool star::VolumeDeformFileFetch::FetchKeypoint(size_t cam_idx, size_t frame_idx, cv::Mat &keypoints,
                                                cv::Mat &descriptors, KeyPointType keypoint_type)
{
    path file_path = FileNameStar(cam_idx, frame_idx, FileType::keypoint_file);
    // Read the file
    if (keypoint_type == KeyPointType::SuperPoints)
    {
        // Read xml using opencv
        cv::FileStorage fs(file_path.string(), cv::FileStorage::READ);
        fs["superpoint_keypoints"] >> keypoints;
        fs["superpoint_descriptors"] >> descriptors;
    }
    else if (keypoint_type == KeyPointType::R2D2)
    {
        // Read xml using opencv
        cv::FileStorage fs(file_path.string(), cv::FileStorage::READ);
        fs["r2d2_keypoints"] >> keypoints;
        fs["r2d2_descriptors"] >> descriptors;
        // Drop the last channel for keypoints
        keypoints = keypoints.colRange(0, 2);
    }
    else if (keypoint_type == KeyPointType::ORB)
    {
        // Read xml using opencv
        cv::FileStorage fs(file_path.string(), cv::FileStorage::READ);
        fs["points"] >> keypoints;
        fs["descriptions"] >> descriptors;
        // Drop the last channel for keypoints
        keypoints = keypoints.colRange(0, 2);
    }
    std::cout << "Keypoints loaded from " << file_path.string() << " !" << std::endl;
    return true;
}

boost::filesystem::path star::VolumeDeformFileFetch::FileNameVolumeDeform(size_t cam_idx, size_t frame_idx, FileType file_type) const
{
    // Construct the file_name
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
        file_name += ".of.png";
        break;
    case FileType::point_cloud_file:
        file_name += ".pcd";
        break;
    case FileType::seg_img_file:
        file_name += ".seg.png";
        break;
    case FileType::keypoint_file:
        file_name += ".orb.xml";  // TODO: temporary using orb feature as feature name
        break;
    default:
        printf("File type is not supported.\n");
        assert(false);
        break;
    }

    // Construct the path
    path file_path = m_data_path / path(file_name);
    return file_path;
}

boost::filesystem::path star::VolumeDeformFileFetch::FileNameStar(size_t cam_idx, size_t frame_idx, FileType file_type) const
{
    return FileNameVolumeDeform(cam_idx, frame_idx, file_type);
}