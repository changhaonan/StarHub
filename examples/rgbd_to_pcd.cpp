/**
 * @file rgbd_to_pcd.cpp
 * @author Haonan Chang (chnme40cs@gmail.com)
 * @brief Load RGBD images and save them as PCD files
 * @version 0.1
 * @date 2022-10-04
 *
 * @copyright Copyright (c) 2022
 *
 */
#include <iostream>
#include <opencv2/opencv.hpp>
#include <star/io/VolumeDeformFileFetch.h>
using namespace star;

int main(int argc, char **argv)
{
    // 1. Preparation
    std::string file_root_path = "/home/robot-learning/Projects/StarHub/data/move_dragon";
    VolumeDeformFileFetch::Ptr file_handler = std::make_shared<VolumeDeformFileFetch>(file_root_path);

    // 2. Load a rgb image && depth image
    cv::Mat color_img;
    file_handler->FetchRGBImage(0, 0, color_img);

    cv::Mat depth_img;
    file_handler->FetchDepthImage(0, 0, depth_img);

    // 3. Create point cloud from rgb & depth image
    cv::imshow("test", color_img);
    cv::waitKey(0);

    return 0;
}