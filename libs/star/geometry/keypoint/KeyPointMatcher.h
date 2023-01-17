#pragma once
#include "KeyPoints.h"
#include <opencv2/features2d.hpp>

namespace star
{

    /**
     * @brief Match keypoints from two images
     * @note Currently, we uses the brute-force matching method from opencv
     * @return matches: (query_idx, train_idx)
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
     * @return matches: (query_idx, train_idx)
     */
    void MatchKeyPointsBFOpenCVHostOnly(
        const cv::Mat &keypoints_query,
        const cv::Mat &keypoints_train,
        const cv::Mat &descriptors_query,
        const cv::Mat &descriptors_train,
        GArraySlice<int2> matches,
        unsigned &num_valid_match,
        const float ratio_thresh,
        const float dist_thresh,
        cudaStream_t stream);

    /**
     * @brief Match keypoints from two images on host only
     * @note Currently, we uses the brute-force matching method from opencv
     * @return matches: (query_idx, train_idx)
     */
    void MatchKeyPointsBFOpenCVHostOnly(
        const cv::Mat &keypoints_query,
        const cv::Mat &keypoints_train,
        const cv::Mat &descriptors_query,
        const cv::Mat &descriptors_train,
        GArrayView<float4> vertex_confid_query,
        GArrayView<float4> vertex_confid_train,
        GArraySlice<int2> matches,
        unsigned &num_valid_match,
        const float ratio_thresh,
        const float pixel_dist_thresh,
        const float vertex_dist_thresh,
        cudaStream_t stream);
}