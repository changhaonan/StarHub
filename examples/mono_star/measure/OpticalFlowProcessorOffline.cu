#include <mono_star/measure/OpticalFlowProcessorOffline.h>
#include <star/img_proc/generate_maps.h>
#include <star/geometry/node_graph/node_graph_opticalflow.h>
#include <star/common/data_transfer.h>
#include <star/visualization/Visualizer.h>
#include <device_launch_parameters.h>

namespace star::device
{
    __global__ void OpticalFlowFilterKernelV2(
        const float2 *__restrict__ opticalflow_raw,
        cudaSurfaceObject_t opticalflow_filtered,
        const float opticalflow_suppress_threshold,
        const unsigned img_cols,
        const unsigned img_rows)
    {
        auto idx = threadIdx.x + blockDim.x * blockIdx.x;
        auto idy = threadIdx.y + blockDim.y * blockIdx.y;
        if (idx >= img_cols || idy >= img_rows)
            return;
        auto tidx = idy * img_cols + idx;
        float2 opticalflow = opticalflow_raw[tidx];
        // Small optical flow will be removed
        if ((opticalflow.x * opticalflow.x + opticalflow.y * opticalflow.y) < opticalflow_suppress_threshold)
        {
            opticalflow = make_float2(0.f, 0.f);
        }
        surf2Dwrite(opticalflow, opticalflow_filtered, idx * sizeof(float2), idy);
    }
}

star::OpticalFlowProcessorOffline::OpticalFlowProcessorOffline()
{
    auto &config = ConfigParser::Instance();

    // Reader-related
    m_fetcher = std::make_shared<VolumeDeformFileFetch>(config.data_path());
    m_start_frame_idx = config.start_frame_idx();
    m_step_frame = config.step_frame();

    // Vis
    m_enable_vis = config.enable_vis();
    m_pcd_size = config.pcd_size();

    // Camera setting
    STAR_CHECK_EQ(config.num_cam(), 1); // Asset mono camera
    m_raw_img_col = config.raw_img_cols();
    m_raw_img_row = config.raw_img_rows();
    m_downsample_img_col = config.downsample_img_cols();
    m_downsample_img_row = config.downsample_img_rows();
    m_cam2world = config.extrinsic()[0];
    m_intrinsic = config.rgb_intrinsic_downsample();
    m_downsample_scale = config.downsample_scale();

    unsigned num_raw_pixel = config.raw_img_cols() * config.raw_img_rows();
    // AllocateBuffer
    cudaSafeCall(cudaMallocHost(
        (void **)&m_raw_opticalflow_img_buff, num_raw_pixel * sizeof(float2)));
    m_g_raw_opticalflow_img.create(num_raw_pixel);
    m_surfel_motion.AllocateBuffer(d_max_num_surfels);
    m_opticalflow_suppress_threshold = config.opticalflow_suppress_threshold();

    // Create texture/surface
    createFloat2TextureSurface(m_downsample_img_row, m_downsample_img_col, m_opticalflow);
}

star::OpticalFlowProcessorOffline::~OpticalFlowProcessorOffline()
{
    cudaSafeCall(cudaFreeHost(m_raw_opticalflow_img_buff));
    m_surfel_motion.ReleaseBuffer();
    m_g_raw_opticalflow_img.release();
    releaseTextureCollect(m_opticalflow);
}

void star::OpticalFlowProcessorOffline::ProcessFrame(
    const SurfelMapTex &surfel_map_this,
    const SurfelMapTex &surfel_map_prev,
    const unsigned frame_idx,
    cudaStream_t stream)
{
    if (frame_idx > 0)
    {
        // OpticalFlow from frame-1
        loadOpticalFlow(frame_idx, stream);

        // Save opticalflow to texture
        createScaledOpticalFlowMap(
            m_g_raw_opticalflow_img.ptr(),
            m_raw_img_row, m_raw_img_col,
            m_downsample_scale,
            m_opticalflow.surface,
            stream);

        // Compute surfel motion
        computeSurfelFlowVisible(surfel_map_prev, surfel_map_this, stream);

        if (m_enable_vis)
        {
            saveContext(frame_idx, stream);
        }
    }
}

void star::OpticalFlowProcessorOffline::loadOpticalFlow(
    const unsigned frame_idx, cudaStream_t stream)
{
    // Load of image
    const auto image_idx = size_t(frame_idx) * m_step_frame + m_start_frame_idx;
    m_fetcher->FetchOFImage(0, image_idx, m_raw_opticalflow_img);

    // Copy to buffer
    memcpy(m_raw_opticalflow_img_buff, m_raw_opticalflow_img.data, sizeof(float2) * m_raw_opticalflow_img.total());
    cudaSafeCall(cudaMemcpyAsync(
        m_g_raw_opticalflow_img.ptr(),
        m_raw_opticalflow_img_buff,
        sizeof(float2) * m_raw_opticalflow_img.total(),
        cudaMemcpyHostToDevice,
        stream));
}

void star::OpticalFlowProcessorOffline::saveContext(const unsigned frame_idx, cudaStream_t stream)
{
    auto &context = easy3d::Context::Instance();
    std::string optical_name = "of";
    // Save images
    context.addImage(optical_name, optical_name);
    visualize::SaveOpticalFlowMap(
        m_opticalflow.texture,
        context.at(optical_name));
}

void star::OpticalFlowProcessorOffline::computeSurfelFlowVisible(
    const SurfelMapTex &surfel_map_prev,
    const SurfelMapTex &surfel_map_this,
    cudaStream_t stream)
{
    // Reset surfel motion
    cudaSafeCall(cudaMemsetAsync(m_surfel_motion.Ptr(), 0, sizeof(float4) * m_surfel_motion.ArraySize(), stream));
    m_surfel_motion.ResizeArrayOrException(surfel_map_prev.num_valid_surfel);

    // Compute surfel motion
    EstimateSurfelMotionFromOpticalFlow(
        surfel_map_prev.vertex_confid,
        surfel_map_this.vertex_confid,
        surfel_map_prev.rgbd,
        surfel_map_this.rgbd,
        m_opticalflow.texture,
        surfel_map_prev.index,
        m_surfel_motion.Slice(),
        m_cam2world,
        m_intrinsic,
        stream);
}