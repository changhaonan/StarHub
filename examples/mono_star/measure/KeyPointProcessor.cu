#include "KeyPointProcessor.h"
#include <star/visualization/Visualizer.h>

namespace star::device
{
    __global__ void Build3DKeyPointKernel(
        cudaTextureObject_t vertex_confid_map,
        cudaTextureObject_t index_map,
        const float2 *__restrict__ keypoints_2d,
        float4* __restrict__ keypoints_3d,
        const unsigned num_keypoints)
    {
        const unsigned idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx >= num_keypoints)
            return;

        const float2 keypoint_2d = keypoints_2d[idx];
        auto index = tex2D<unsigned>(index_map, keypoint_2d.x, keypoint_2d.y);
        if (index) {
            auto vertex = tex2D<float4>(vertex_confid_map, keypoint_2d.x, keypoint_2d.y);
            keypoints_3d[idx] = vertex;
        }
        else {
            keypoints_3d[idx] = make_float4(0, 0, 0, 0);
        }
    }
}

star::KeyPointProcessor::KeyPointProcessor(KeyPointType keypoint_type) : m_keypoint_type(keypoint_type)
{
    auto &config = ConfigParser::Instance();

    m_fetcher = std::make_shared<VolumeDeformFileFetch>(config.data_path());
    m_keypoints = std::make_shared<KeyPoints>(m_keypoint_type);

    // Camera parameters
    m_step_frame = config.step_frame();
    m_start_frame_idx = config.start_frame_idx();

    // Create host buffer
    cudaSafeCall(cudaMallocHost(&m_keypoint_buffer, sizeof(float) * 2 * d_max_num_keypoints));
    cudaSafeCall(cudaMallocHost(&m_descriptor_buffer, sizeof(float) * d_max_num_keypoints * m_keypoints->DescriptorDim()));

    m_g_keypoint_2d.AllocateBuffer(d_max_num_keypoints);

    m_pcd_size = config.pcd_size();
}

star::KeyPointProcessor::~KeyPointProcessor()
{
    cudaSafeCall(cudaFreeHost(m_keypoint_buffer));
    cudaSafeCall(cudaFreeHost(m_descriptor_buffer));

    m_g_keypoint_2d.ReleaseBuffer();
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
        m_keypoints->Descriptor().Ptr(),
        m_descriptor_buffer,
        m_descriptor_mat.total() * sizeof(float),
        cudaMemcpyHostToDevice,
        stream));

    // Resize
    m_g_keypoint_2d.ResizeArrayOrException(num_keypoints_detected);
    m_keypoints->Resize(num_keypoints_detected);
    // Build 3d keypoint
    build3DKeyPoint(surfel_map_tex, num_keypoints_detected, stream);

    // Sync stream
    cudaSafeCall(cudaStreamSynchronize(stream));

    // Save context
    saveContext(frame_idx, stream);
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
        surfel_map_tex.index,
        m_g_keypoint_2d.Ptr(),
        m_keypoints->VertexConfid().Ptr(),
        num_keypoints);
    m_keypoints->Resize(num_keypoints);
}

void star::KeyPointProcessor::saveContext(unsigned frame_idx, cudaStream_t stream)
{
    auto &context = easy3d::Context::Instance();

    context.addPointCloud("keypoints", "", Eigen::Matrix4f::Identity(), m_pcd_size);
    visualize::SavePointCloud(
        m_keypoints->VertexConfidReadOnly(), context.at("keypoints")
    );
    
}