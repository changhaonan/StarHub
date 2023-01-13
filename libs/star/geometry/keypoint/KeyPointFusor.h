#pragma once
#include "KeyPoints.h"
#include <star/common/algorithm_types.h>

namespace star
{
    // Fuse new key points into the existing key points.
    class KeyPointFusor
    {
    public:
        using Ptr = std::shared_ptr<KeyPointFusor>;
        KeyPointFusor(
            const float kp_match_threshold);
        ~KeyPointFusor();

        /* Compare the old keypoints and new keypoints, if two kp are matched and distance is
         * less than a threshold, replace the old descriptor with the new one. if new keypoints
         * has no match, then it is regarded as new keypoints, and append to the end of the tar keypoints.
         */
        void Fuse(
            KeyPoints::Ptr old_keypoints,
            KeyPoints::Ptr new_keypoints,
            GArrayView<int2> matches,
            const bool enable_append,
            cudaStream_t stream);

    private:
        void resetIdicator(
            const unsigned num_new_keypoints,
            cudaStream_t stream
        );
        // Mark if new kp has a match and replace the old descriptor with the newly matched one
        void markMatchAndReplace(
            KeyPoints::Ptr old_keypoints,
            KeyPoints::Ptr new_keypoints,
            GArrayView<int2> matches,
            cudaStream_t stream);
        // Append those new keypoints that has no match
        void appendNewKeyPoints(
            KeyPoints::Ptr old_keypoints,
            KeyPoints::Ptr new_keypoints,
            cudaStream_t stream);

        float m_kp_match_threshold;
        GBufferArray<unsigned> m_not_matched_indicator;
        PrefixSum m_not_matched_prefix_sum;
        unsigned *m_host_num_not_matches;   // Host
    };

}