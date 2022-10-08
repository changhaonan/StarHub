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
#include <star/geometry/geometry_map/SurfelMap.h>
#include <star/geometry/geometry_map/SurfelMapInitializer.h>

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

    // 3. Parameter
    unsigned width = 640;
    unsigned height = 480;
    float clip_near = 0.1f;
    float clip_far = 10.f;
    float surfel_radius_scale = 1.f;
    float focal_x = 520.f;
    float focal_y = 520.f;
    float principal_x = 320.f;
    float principal_y = 240.f;
    Intrinsic intrinsic(
        focal_x, focal_y, principal_x, principal_y
    );

    // 4. Create initializer
    auto surfel_map = std::make_shared<SurfelMap>(width, height);
    SurfelMapInitializer surfel_map_initializer(
        width, height, clip_near, clip_far, surfel_radius_scale, intrinsic
    );

    return 0;
}