#include "KeyPointDetectProcessor.h"
#include <star/geometry/surfel/SurfelGeometryInitializer.h>
#include <star/geometry/keypoint/KeyPointMatcher.h>
#include <star/visualization/Visualizer.h>

// namespace star::device
// {
//     __global__ void GetMatchedPointKernel(
//         const float4 *__restrict__ vertex_confid_src,
//         const float4 *__restrict__ vertex_confid_dst,
//         float4* __restrict__ matched_vertex_confid_src,
//         float4* __restrict__ matched_vertex_confid_dst,
//         int2 *__restrict__ matches,
//         const unsigned num_matches)
//     {
//         const unsigned idx = threadIdx.x + blockIdx.x * blockDim.x;
//         if (idx >= num_matches)
//             return;

//         const int2 match = matches[idx];
//         matched_vertex_confid_src[idx] = vertex_confid_src[match.x];
//         matched_vertex_confid_dst[idx] = vertex_confid_dst[match.y];
//     }

// }

star::KeyPointDetectProcessor::KeyPointDetectProcessor()
    : m_buffer_idx(0), m_num_valid_matches(0)
{
    auto &config = ConfigParser::Instance();
    m_keypoint_type = config.keypoint_type();

    m_fetcher = std::make_shared<VolumeDeformFileFetch>(config.data_path());

    // Camera parameters
    m_step_frame = config.step_frame();
    m_start_frame_idx = config.start_frame_idx();
    m_cam2world = config.extrinsic()[0];
    m_enable_semantic_surfel = config.enable_semantic_surfel();

    // Create host buffer
    cudaSafeCall(cudaMallocHost(&m_keypoint_buffer, sizeof(float) * 2 * d_max_num_keypoints));
    cudaSafeCall(cudaMallocHost(&m_descriptor_buffer, sizeof(float) * d_max_num_keypoints * KeyPoints::GetDescriptorDim(m_keypoint_type)));

    m_g_keypoints.AllocateBuffer(d_max_num_keypoints);
    m_g_descriptors.AllocateBuffer(d_max_num_keypoints * KeyPoints::GetDescriptorDim(m_keypoint_type));
    m_keypoint_matches.AllocateBuffer(d_max_num_keypoints);

    m_detected_keypoints = std::make_shared<star::KeyPoints>(config.keypoint_type());

    m_matched_vertex_src.AllocateBuffer(d_max_num_keypoints);
    m_matched_vertex_dst.AllocateBuffer(d_max_num_keypoints);

    // Matching related
    m_kp_match_ratio_thresh = config.kp_match_ratio_thresh();
    m_kp_match_dist_thresh = config.kp_match_dist_thresh();

    // Vis
    m_enable_vis = config.enable_vis();
    m_pcd_size = config.pcd_size();
}

star::KeyPointDetectProcessor::~KeyPointDetectProcessor()
{
    cudaSafeCall(cudaFreeHost(m_keypoint_buffer));
    cudaSafeCall(cudaFreeHost(m_descriptor_buffer));

    m_g_keypoints.ReleaseBuffer();
    m_g_descriptors.ReleaseBuffer();
    m_keypoint_matches.ReleaseBuffer();

    m_matched_vertex_src.ReleaseBuffer();
    m_matched_vertex_dst.ReleaseBuffer();
}

void star::KeyPointDetectProcessor::ProcessFrame(const SurfelMap &surfel_map, unsigned frame_idx, cudaStream_t stream)
{
    const auto image_idx = size_t(frame_idx) * m_step_frame + m_start_frame_idx;
    m_fetcher->FetchKeypoint(0, image_idx, m_keypoint_mat, m_descriptor_mat, m_keypoint_type);

    // Load the keypoint and descriptor into gpu
    unsigned num_keypoints_detected = m_keypoint_mat.rows;
    // CPU copy
    memcpy(m_keypoint_buffer, m_keypoint_mat.data,
           sizeof(float) * m_keypoint_mat.total());
    memcpy(m_descriptor_buffer, m_descriptor_mat.data,
           sizeof(float) * m_descriptor_mat.total());

    // Copy to GPU
    cudaSafeCall(cudaMemcpyAsync(
        m_g_keypoints.Ptr(),
        m_keypoint_buffer,
        m_keypoint_mat.total() * sizeof(float),
        cudaMemcpyHostToDevice,
        stream));
    cudaSafeCall(cudaMemcpyAsync(
        m_g_descriptors.Ptr(),
        m_descriptor_buffer,
        m_descriptor_mat.total() * sizeof(float),
        cudaMemcpyHostToDevice,
        stream));

    // Resize
    m_g_keypoints.ResizeArrayOrException(num_keypoints_detected);
    m_g_descriptors.ResizeArrayOrException(num_keypoints_detected * KeyPoints::GetDescriptorDim(m_keypoint_type));
    
    // Build 3d keypoint
    // Init keypoint geometry
    SurfelGeometryInitializer::InitFromGeometryMap(
        *m_detected_keypoints,
        surfel_map,
        m_g_keypoints.View(),
        m_cam2world,
        m_enable_semantic_surfel,
        stream);

    // if (frame_idx != 0 && m_model_keypoints[m_buffer_idx]->NumKeyPoints() > 0)
    // {
    //     // Apply matching between model keypoints and detected keypoints
    //     MatchKeyPointsBFOpenCV(
    //         *m_model_keypoints[m_buffer_idx],
    //         *m_keypoints_dected,
    //         m_keypoint_matches.Slice(),
    //         m_num_valid_matches,
    //         m_kp_match_ratio_thresh,
    //         m_kp_match_dist_thresh,
    //         stream);
    //     m_keypoint_matches.ResizeArrayOrException(m_num_valid_matches);
    // }

    // Sync stream
    cudaSafeCall(cudaStreamSynchronize(stream));

    // Save context
    if (m_enable_vis)
        saveContext(frame_idx, stream);
}

void star::KeyPointDetectProcessor::saveContext(unsigned frame_idx, cudaStream_t stream)
{
    auto &context = easy3d::Context::Instance();

    context.addPointCloud("d_keypoints", "", Eigen::Matrix4f::Identity(), m_pcd_size);
    visualize::SaveColoredPointCloud(
        m_detected_keypoints->LiveVertexConfidenceReadOnly(), 
        m_detected_keypoints->ColorTimeReadOnly(),
        context.at("d_keypoints"));

    // // Draw the matched keypoints
    // if (m_num_valid_matches > 0) {
    //     getMatchedKeyPoints(stream);
    //     context.addPointCloud("matched_keypoints", "", Eigen::Matrix4f::Identity(), m_pcd_size);
    //     visualize::SaveMatchedPointCloud(
    //         m_matched_vertex_src.View(),
    //         m_matched_vertex_dst.View(),
    //         context.at("matched_keypoints")
    //     );
    // }
}

void star::KeyPointDetectProcessor::getMatchedKeyPoints(cudaStream_t stream)
{
    // // Resize
    // m_matched_vertex_src.ResizeArrayOrException(m_num_valid_matches);
    // m_matched_vertex_dst.ResizeArrayOrException(m_num_valid_matches);
    
    // // Get matched keypoints
    // dim3 blk(128);
    // dim3 grid(divUp(m_num_valid_matches, blk.x));
    // device::GetMatchedPointKernel<<<grid, blk, 0, stream>>>(
    //     m_model_keypoints[m_buffer_idx]->LiveVertexConfidence().Ptr(),
    //     m_keypoints_dected->LiveVertexConfidence().Ptr(),
    //     m_matched_vertex_src.Ptr(),
    //     m_matched_vertex_dst.Ptr(),
    //     m_keypoint_matches.Ptr(),
    //     m_num_valid_matches);
}