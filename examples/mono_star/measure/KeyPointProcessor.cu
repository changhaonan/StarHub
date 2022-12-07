#include "KeyPointProcessor.h"
#include <star/geometry/keypoint/KeyPointMatcher.h>
#include <star/visualization/Visualizer.h>

namespace star::device
{
    __global__ void Build3DKeyPointKernel(
        cudaTextureObject_t vertex_confid_map,
        cudaTextureObject_t normal_radius_map,
        cudaTextureObject_t color_time_map,
        cudaTextureObject_t index_map,
        const float2 *__restrict__ keypoints_2d,
        float4 *__restrict__ kp_vertex_confid,
        float4 *__restrict__ kp_normal_radius,
        float4 *__restrict__ kp_color_time,
        const unsigned num_keypoints)
    {
        const unsigned idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx >= num_keypoints)
            return;

        const float2 keypoint_2d = keypoints_2d[idx];
        auto index = tex2D<unsigned>(index_map, keypoint_2d.x, keypoint_2d.y);
        if (index)
        {
            kp_vertex_confid[idx] = tex2D<float4>(vertex_confid_map, keypoint_2d.x, keypoint_2d.y);
            kp_normal_radius[idx] = tex2D<float4>(normal_radius_map, keypoint_2d.x, keypoint_2d.y);
            kp_color_time[idx] = tex2D<float4>(color_time_map, keypoint_2d.x, keypoint_2d.y);
        }
        else
        {
            kp_vertex_confid[idx] = make_float4(0, 0, 0, 0);
            kp_normal_radius[idx] = make_float4(0, 0, 0, 0);
            kp_color_time[idx] = make_float4(0, 0, 0, 0);
        }
    }

    __global__ void GetMatchedPointKernel(
        const float4 *__restrict__ vertex_confid_src,
        const float4 *__restrict__ vertex_confid_dst,
        float4* __restrict__ matched_vertex_confid_src,
        float4* __restrict__ matched_vertex_confid_dst,
        int2 *__restrict__ matches,
        const unsigned num_matches)
    {
        const unsigned idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx >= num_matches)
            return;

        const int2 match = matches[idx];
        matched_vertex_confid_src[idx] = vertex_confid_src[match.x];
        matched_vertex_confid_dst[idx] = vertex_confid_dst[match.y];
    }

}

star::KeyPointProcessor::KeyPointProcessor()
    : m_buffer_idx(0), m_num_valid_matches(0)
{
    auto &config = ConfigParser::Instance();
    m_keypoint_type = config.keypoint_type();

    m_fetcher = std::make_shared<VolumeDeformFileFetch>(config.data_path());
    m_keypoints_dected = std::make_shared<KeyPoints>(m_keypoint_type);
    m_model_keypoints[0] = std::make_shared<KeyPoints>(m_keypoint_type);
    m_model_keypoints[1] = std::make_shared<KeyPoints>(m_keypoint_type);
    
    // Camera parameters
    m_step_frame = config.step_frame();
    m_start_frame_idx = config.start_frame_idx();

    // Create host buffer
    cudaSafeCall(cudaMallocHost(&m_keypoint_buffer, sizeof(float) * 2 * d_max_num_keypoints));
    cudaSafeCall(cudaMallocHost(&m_descriptor_buffer, sizeof(float) * d_max_num_keypoints * m_keypoints_dected->DescriptorDim()));

    m_g_keypoint_2d.AllocateBuffer(d_max_num_keypoints);
    m_keypoint_matches.AllocateBuffer(d_max_num_keypoints);

    m_matched_vertex_src.AllocateBuffer(d_max_num_keypoints);
    m_matched_vertex_dst.AllocateBuffer(d_max_num_keypoints);

    // Matching related
    m_kp_match_ratio_thresh = config.kp_match_ratio_thresh();
    m_kp_match_dist_thresh = config.kp_match_dist_thresh();

    // Vis
    m_enable_vis = config.enable_vis();
    m_pcd_size = config.pcd_size();
}

star::KeyPointProcessor::~KeyPointProcessor()
{
    cudaSafeCall(cudaFreeHost(m_keypoint_buffer));
    cudaSafeCall(cudaFreeHost(m_descriptor_buffer));

    m_g_keypoint_2d.ReleaseBuffer();
    m_keypoint_matches.ReleaseBuffer();

    m_matched_vertex_src.ReleaseBuffer();
    m_matched_vertex_dst.ReleaseBuffer();
}

void star::KeyPointProcessor::ProcessFrame(const SurfelMapTex &surfel_map_tex, unsigned frame_idx, cudaStream_t stream)
{
    const auto image_idx = size_t(frame_idx) * m_step_frame + m_start_frame_idx;
    m_fetcher->FetchKeypoint(0, image_idx, m_keypoint_2d_mat, m_descriptor_mat, m_keypoint_type);

    // Load the keypoint and descriptor into gpu
    unsigned num_keypoints_detected = m_keypoint_2d_mat.rows;
    // CPU copy
    memcpy(m_keypoint_buffer, m_keypoint_2d_mat.data,
           sizeof(float) * m_keypoint_2d_mat.total());
    memcpy(m_descriptor_buffer, m_descriptor_mat.data,
           sizeof(float) * m_descriptor_mat.total());

    // Copy to GPU
    cudaSafeCall(cudaMemcpyAsync(
        m_g_keypoint_2d.Ptr(),
        m_keypoint_buffer,
        m_keypoint_2d_mat.total() * sizeof(float),
        cudaMemcpyHostToDevice,
        stream));
    cudaSafeCall(cudaMemcpyAsync(
        m_keypoints_dected->Descriptor().Ptr(),
        m_descriptor_buffer,
        m_descriptor_mat.total() * sizeof(float),
        cudaMemcpyHostToDevice,
        stream));

    // Resize
    m_g_keypoint_2d.ResizeArrayOrException(num_keypoints_detected);
    m_keypoints_dected->Resize(num_keypoints_detected);
    // Build 3d keypoint
    build3DKeyPoint(surfel_map_tex, num_keypoints_detected, stream);

    if (frame_idx != 0 && m_model_keypoints[m_buffer_idx]->NumKeyPoints() > 0)
    {
        // Apply matching between model keypoints and detected keypoints
        MatchKeyPointsBFOpenCV(
            *m_model_keypoints[m_buffer_idx],
            *m_keypoints_dected,
            m_keypoint_matches.Slice(),
            m_num_valid_matches,
            m_kp_match_ratio_thresh,
            m_kp_match_dist_thresh,
            stream);
        m_keypoint_matches.ResizeArrayOrException(m_num_valid_matches);
    }

    // Sync stream
    cudaSafeCall(cudaStreamSynchronize(stream));

    // Save context
    if (m_enable_vis)
        saveContext(frame_idx, stream);

    // Update model keypoints
    updateModelKeyPoints(frame_idx, stream);
}

void star::KeyPointProcessor::build3DKeyPoint(
    const SurfelMapTex &surfel_map_tex,
    unsigned num_keypoints,
    cudaStream_t stream)
{
    const dim3 block(256);
    const dim3 grid(divUp(num_keypoints, block.x));
    device::Build3DKeyPointKernel<<<grid, block, 0, stream>>>(
        surfel_map_tex.vertex_confid,
        surfel_map_tex.normal_radius,
        surfel_map_tex.color_time,
        surfel_map_tex.index,
        m_g_keypoint_2d.Ptr(),
        m_keypoints_dected->LiveVertexConfidence().Ptr(),
        m_keypoints_dected->LiveNormalRadius().Ptr(),
        m_keypoints_dected->ColorTime().Ptr(),
        num_keypoints);
    m_keypoints_dected->Resize(num_keypoints);
}

void star::KeyPointProcessor::updateModelKeyPoints(
    const unsigned frame_idx,
    cudaStream_t stream)
{
    m_model_keypoints[m_buffer_idx].swap(m_keypoints_dected);
}

void star::KeyPointProcessor::saveContext(unsigned frame_idx, cudaStream_t stream)
{
    auto &context = easy3d::Context::Instance();

    context.addPointCloud("keypoints", "", Eigen::Matrix4f::Identity(), m_pcd_size);
    visualize::SaveColoredPointCloud(
        m_keypoints_dected->LiveVertexConfidenceReadOnly(), 
        m_keypoints_dected->ColorTimeReadOnly(),
        context.at("keypoints"));

    // Draw the matched keypoints
    if (m_num_valid_matches > 0) {
        getMatchedKeyPoints(stream);
        context.addPointCloud("matched_keypoints", "", Eigen::Matrix4f::Identity(), m_pcd_size);
        visualize::SaveMatchedPointCloud(
            m_matched_vertex_src.View(),
            m_matched_vertex_dst.View(),
            context.at("matched_keypoints")
        );
    }
}

void star::KeyPointProcessor::getMatchedKeyPoints(cudaStream_t stream)
{
    // Resize
    m_matched_vertex_src.ResizeArrayOrException(m_num_valid_matches);
    m_matched_vertex_dst.ResizeArrayOrException(m_num_valid_matches);
    
    // Get matched keypoints
    dim3 blk(128);
    dim3 grid(divUp(m_num_valid_matches, blk.x));
    device::GetMatchedPointKernel<<<grid, blk, 0, stream>>>(
        m_model_keypoints[m_buffer_idx]->LiveVertexConfidence().Ptr(),
        m_keypoints_dected->LiveVertexConfidence().Ptr(),
        m_matched_vertex_src.Ptr(),
        m_matched_vertex_dst.Ptr(),
        m_keypoint_matches.Ptr(),
        m_num_valid_matches);
}