#pragma once
#include <star/common/common_types.h>
#include <star/io/VolumeDeformFileFetch.h>
#include <star/geometry/keypoint/KeyPoints.h>
#include <star/geometry/geometry_map/SurfelMap.h>
#include <mono_star/common/ConfigParser.h>
#include <opencv2/features2d.hpp>
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
            const SurfelMapTex &measure_surfel_map,
            const SurfelMapTex &model_surfel_map,
            const unsigned frame_idx,
            cudaStream_t stream);
        // Fetch-API
        GArrayView<float2> Get2DKeyPointsReadOnly() const { return m_g_keypoints.View(); }
        GArrayView<unsigned char> MeasureDescriptorsReadOnly() const { return m_measure_keypoints->DescriptorReadOnly(); }
        star::KeyPoints::Ptr MeasureKeyPoints() const { return m_measure_keypoints; }
        star::KeyPoints::Ptr ModelKeyPoints() const { return m_model_keypoints; }
        GArrayView<int2> GetMatchedKeyPointsReadOnly() const { return m_keypoint_matches.View(); }

        // Build Solver
        // KeyPoint4Solver GenerateKeyPoint4Solver(
        //     const SurfelMapTex &measure_surfel_map,
        //     const SurfelMapTex &model_surfel_map,
        //     const SurfelGeometry& model_geometry,
        //     const Renderer::SolverMaps &solver_maps,
        //     const unsigned frame_idx,
        //     cudaStream_t stream);

    private:
        void getMatchedKeyPoints(
            const KeyPoints &keypoints_src,
            const KeyPoints &keypoints_dst,
            cudaStream_t stream);
        void saveContext(
            unsigned frame_idx,
            cudaStream_t stream);
        // KeyPoint Online Detection
        void detectFeature(
            const SurfelMapTex &surfel_map,
            cv::Mat &keypoint,
            cv::Mat &descriptor,
            cudaStream_t stream);
        void detectORBFeature(
            cudaTextureObject_t rgbd_tex,
            cv::Mat &keypoint,
            cv::Mat &descriptor,
            cudaStream_t stream);
        // Build keypoints in world coordinate
        void buildKeyPoints(
            const SurfelMapTex &surfel_map,
            const cv::Mat &keypoint,
            const cv::Mat &descriptor,
            KeyPoints::Ptr keypoints,
            const Extrinsic& T_to_world,
            cudaStream_t stream);

        // Camera parameters
        unsigned m_step_frame;
        unsigned m_start_frame_idx;
        unsigned m_buffer_idx;
        Extrinsic m_cam2world;
        bool m_enable_semantic_surfel;
        float m_downsample_scale;
        unsigned m_image_width;
        unsigned m_image_height;

        // Matching-related
        float m_kp_match_ratio_thresh;
        float m_kp_match_dist_thresh;
        unsigned m_num_valid_matches;
        GBufferArray<int2> m_keypoint_matches;

        // Keypoint
        KeyPoints::Ptr m_measure_keypoints;
        KeyPoints::Ptr m_model_keypoints;

        // Buffer
        cv::Mat m_keypoint_src; // Model
        cv::Mat m_descriptor_src;
        cv::Mat m_keypoint_tar; // Measure
        cv::Mat m_descriptor_tar;
        void *m_keypoint_buffer;
        void *m_descriptor_buffer;
        // Buffer for detect
        GArray<uchar3> m_g_rgb;
        uchar3 *m_h_rgb;
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