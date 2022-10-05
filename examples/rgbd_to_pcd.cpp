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
// #include <argparse/argparse.hpp>
#include <iostream>
#include <Eigen/Core>

int main(int argc, char **argv)
{
    auto M = Eigen::Matrix3d::Random();
    std::cout << M << std::endl;
    return 0;
}