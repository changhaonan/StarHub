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
     * @brief KeyPointProcessor
     * @note KeyPointProcessor is kind of similar to DynamicGeometry processor,
     * providing update and matching keypoints.
     */
    class KeyPointProcessor
    {
    public:
        using Ptr = std::shared_ptr<KeyPointProcessor>;
        STAR_NO_COPY_ASSIGN_MOVE(KeyPointProcessor);
        KeyPointProcessor();
        ~KeyPointProcessor();
        void ProcessFrame(
            const SurfelMapTex &surfel_map_tex,
            const unsigned frame_idx,
            cudaStream_t stream);

    private:
        void build3DKeyPoint(
            const SurfelMapTex &surfel_map_tex,
            unsigned num_keypoints,
            cudaStream_t stream);
        void updateModelKeyPoints(
            const unsigned frame_idx,
            cudaStream_t stream);
        void matchKeyPoints(
            cudaStream_t stream);
        void getMatchedKeyPoints(
            cudaStream_t stream);
        void saveContext(
            unsigned frame_idx,
            cudaStream_t stream);

        // Camera parameters
        unsigned m_step_frame;
        unsigned m_start_frame_idx;
        unsigned m_buffer_idx;

        // Matching-related
        float m_kp_match_ratio_thresh;
        float m_kp_match_dist_thresh;
        unsigned m_num_valid_matches;
        GBufferArray<int2> m_keypoint_matches;

        // Buffer
        cv::Mat m_keypoint_2d_mat;
        cv::Mat m_descriptor_mat;
        void *m_keypoint_buffer;
        void *m_descriptor_buffer;
        // Buffer for vis
        GBufferArray<float4> m_matched_vertex_src;
        GBufferArray<float4> m_matched_vertex_dst;

        GBufferArray<float2> m_g_keypoint_2d;
        KeyPointType m_keypoint_type;
        KeyPoints::Ptr m_keypoints_dected;
        KeyPoints::Ptr m_model_keypoints[2];
        VolumeDeformFileFetch::Ptr m_fetcher;

        // Vis
        bool m_enable_vis;
        float m_pcd_size;
    };

}