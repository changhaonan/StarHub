#pragma once
#include <star/common/common_types.h>
#include <star/io/VolumeDeformFileFetch.h>
#include <star/geometry/keypoint/KeyPoints.h>
#include <star/geometry/geometry_map/SurfelMap.h>
#include <mono_star/common/ConfigParser.h>
// Viewer
#include <easy3d_viewer/context.hpp>

namespace star
{
    /**
     * @brief KeyPointDetectProcessor
     * @note KeyPointDetectProcessor is kind of similar to DynamicGeometry processor,
     * providing update and matching keypoints.
     */
    class KeyPointDetectProcessor
    {
    public:
        using Ptr = std::shared_ptr<KeyPointDetectProcessor>;
        STAR_NO_COPY_ASSIGN_MOVE(KeyPointDetectProcessor);
        KeyPointDetectProcessor();
        ~KeyPointDetectProcessor();
        void ProcessFrame(
            const SurfelMap &surfel_map,
            const KeyPoints &model_keypoints,
            const unsigned frame_idx,
            cudaStream_t stream);
        // Fetch-API
        GArrayView<float2> Get2DKeyPointsReadOnly() const { return m_g_keypoints.View(); }
        GArrayView<float> GetDescriptorsReadOnly() const { return m_detected_keypoints->DescriptorReadOnly(); }
        star::KeyPoints::Ptr GetKeyPointsReadOnly() const { return m_detected_keypoints; }
        GArrayView<int2> GetMatchedKeyPointsReadOnly() const { return m_keypoint_matches.View(); }
    private:
        void matchKeyPoints(
            cudaStream_t stream);
        void getMatchedKeyPoints(
            const KeyPoints& keypoints_src,
            const KeyPoints& keypoints_dst,
            cudaStream_t stream);
        void saveContext(
            const KeyPoints& model_keypoints,
            unsigned frame_idx,
            cudaStream_t stream);

        // Camera parameters
        unsigned m_step_frame;
        unsigned m_start_frame_idx;
        unsigned m_buffer_idx;
        Extrinsic m_cam2world;
        bool m_enable_semantic_surfel;
        float m_downsample_scale;

        // Matching-related
        float m_kp_match_ratio_thresh;
        float m_kp_match_dist_thresh;
        unsigned m_num_valid_matches;
        GBufferArray<int2> m_keypoint_matches;

        // Keypoint
        KeyPoints::Ptr m_detected_keypoints;

        // Buffer
        cv::Mat m_keypoint_mat;
        cv::Mat m_descriptor_mat;
        void *m_keypoint_buffer;
        void *m_descriptor_buffer;
        // Buffer for vis
        GBufferArray<float4> m_matched_vertex_src;
        GBufferArray<float4> m_matched_vertex_dst;

        GBufferArray<float2> m_g_keypoints;
        KeyPointType m_keypoint_type;
        unsigned m_dim_descriptor;
        VolumeDeformFileFetch::Ptr m_fetcher;

        // Vis
        bool m_enable_vis;
        float m_pcd_size;
    };

}