#pragma once
#include "KeyPoints.h"
#include <opencv2/features2d.hpp>

namespace star {

    /**
     * @brief Match keypoints from two images
     * @note Currently, we uses the brute-force matching method from opencv
    */
    void MatchKeyPointsBFOpenCV(
        const KeyPoints& key_points_src,
        const KeyPoints& key_points_dst,
        GArraySlice<int2> matches,
        unsigned& num_valid_match,
        const float ratio_thresh,
        const float dist_thresh,
        cudaStream_t stream
    );
}