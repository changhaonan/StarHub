#pragma once
#include <opencv2/opencv.hpp>

namespace star
{
    class RGBDImage
    {
    public:
        RGBDImage(const cv::Mat &rgb_image, const cv::Mat &depth_image);

    private:
        cv::Mat m_rgb_image;
        cv::Mat m_depth_image;
    }
} // namespace star