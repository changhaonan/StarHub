#pragma once
#include "KeyPoints.h"
#include <opencv2/features2d.hpp>

namespace star
{

    /**
     * @brief Match keypoints from two images
     * @note Currently, we uses the brute-force matching method from opencv
     */
    void MatchKeyPointsBFOpenCV(
        const KeyPoints &keypoints_query,
        const KeyPoints &keypoints_train,
        GArraySlice<int2> matches,
        unsigned &num_valid_match,
        const float ratio_thresh,
        const float dist_thresh,
        cudaStream_t stream);

    /**
     * @brief Match keypoints from two images on host only
     * @note Currently, we uses the brute-force matching method from opencv
     */
    void MatchKeyPointsBFOpenCVHostOnly(
        const cv::Mat& keypoints_query,
        const cv::Mat& keypoints_train,
        const cv::Mat& descriptors_query,
        const cv::Mat& descriptors_train,
        GArraySlice<int2> matches,
        unsigned &num_valid_match,
        const float ratio_thresh,
        const float dist_thresh,
        cudaStream_t stream);

}