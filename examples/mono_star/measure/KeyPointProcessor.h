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
    class KeyPointProcessor
    {
    public:
        using Ptr = std::shared_ptr<KeyPointProcessor>;
        STAR_NO_COPY_ASSIGN_MOVE(KeyPointProcessor);
        KeyPointProcessor(KeyPointType keypoint_type);
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
        void saveContext(
            const unsigned frame_idx,
            cudaStream_t stream);

        // Camera parameters
        unsigned m_step_frame;
        unsigned m_start_frame_idx;

        // Buffer
        cv::Mat m_keypoint_2d_mat;
        cv::Mat m_descriptor_mat;
        void* m_keypoint_buffer;
        void* m_descriptor_buffer;
        
        GBufferArray<float2> m_g_keypoint_2d;
        KeyPointType m_keypoint_type;
        KeyPoints::Ptr m_keypoints;
        VolumeDeformFileFetch::Ptr m_fetcher;

        // Vis
        float m_pcd_size;
    };

}