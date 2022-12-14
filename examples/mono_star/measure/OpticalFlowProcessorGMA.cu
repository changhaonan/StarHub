#include <mono_star/measure/OpticalFlowProcessorGMA.h>
#include <star/geometry/node_graph/node_graph_opticalflow.h>
#include <star/common/data_transfer.h>
#include <device_launch_parameters.h>

namespace star::device
{
    __global__ void loadRGBWithBackgroundKernel(
        cudaTextureObject_t texture_rgbd,
        float4 *__restrict__ img_rgbd,
        const uchar3 *__restrict__ img_background,
        const unsigned img_cols,
        const unsigned img_rows)
    {
        auto idx = threadIdx.x + blockDim.x * blockIdx.x;
        auto idy = threadIdx.y + blockDim.y * blockIdx.y;
        if (idx >= img_cols || idy >= img_rows)
            return;
        float4 rgbd = tex2D<float4>(texture_rgbd, idx, idy);
        unsigned pidx = idy * img_cols + idx;
        if (rgbd.w == 0.f)
        { // The depth channel is invalid no matter what reason
            uchar3 color_bg = img_background[pidx];
            img_rgbd[pidx] = make_float4(
                float(color_bg.x) / 127.5f - 1.f,
                float(color_bg.y) / 127.5f - 1.f,
                float(color_bg.z) / 127.5f - 1.f,
                0.f);
        }
        else
        {
            img_rgbd[pidx] = rgbd;
        }
    }

    __global__ void OpticalFlowFilterKernel(
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

star::OpticalFlowProcessorGMA::OpticalFlowProcessorGMA()
{
    auto &config = ConfigParser::Instance();
    std::string device_string = config.nn_device();
    std::string model_path;
    if (device_string == "cpu")
    {
        model_path = config.model_path() + "/traced_gma_model_cpu.pt";
    }
    else if (device_string == "cuda:0")
    {
        model_path = config.model_path() + "/traced_gma_model_gpu.pt";
    }
    else
    {
        std::cerr << "device not implemented" << std::endl;
        assert(false);
    }
    m_model = std::make_shared<nn::OpticalFlowModel>();
    m_model->Load(model_path);
    m_model->SetDevice(device_string); // or "cuda:0"

    // Vis
    m_enable_vis = config.enable_vis();
    m_pcd_size = config.pcd_size();

    // Camera setting
    m_num_cam = config.num_cam();
    STAR_CHECK_EQ(m_num_cam, 1); // Asset mono camera
    m_downsample_img_col = config.downsample_img_cols();
    m_downsample_img_row = config.downsample_img_rows();
    m_cam2world = config.extrinsic()[0];
    m_intrinsic = config.rgb_intrinsic_downsample();

    unsigned num_pixel = m_downsample_img_col * m_downsample_img_row;
    // AllocateBuffer
    m_rgbd_prev.AllocateBuffer(num_pixel);
    m_rgbd_this.AllocateBuffer(num_pixel);
    m_rgbd_prev.ResizeArrayOrException(num_pixel);
    m_rgbd_this.ResizeArrayOrException(num_pixel);
    m_surfel_motion.AllocateBuffer(d_max_num_surfels);

    m_opticalflow_suppress_threshold = config.opticalflow_suppress_threshold();

    // Background things
    m_use_static_background = true;
    std::string img_bg_path = config.model_path() + "/background_dark.png";
    // std::string img_bg_path = config.model_path() + "/background_light.png";
    cv::Mat img_bg = cv::imread(img_bg_path);
    m_background_img.create(m_downsample_img_row, m_downsample_img_col);
    cv::Mat img_bg_resize;
    cv::resize(img_bg, img_bg_resize, cv::Size(m_downsample_img_col, m_downsample_img_row), 0, 0, cv::INTER_CUBIC);
    cudaSafeCall(cudaMemcpy(
        (void *)m_background_img.ptr(), (void *)img_bg_resize.data, sizeof(uchar3) * num_pixel, cudaMemcpyHostToDevice));

    // Create texture/surface
    createFloat2TextureSurface(m_downsample_img_row, m_downsample_img_col, m_opticalflow);

    // Start with 3 load running if the device is not "cpu"
    if (m_model->GetDevice() != "cpu")
        coldStart();
}

star::OpticalFlowProcessorGMA::~OpticalFlowProcessorGMA()
{
    m_rgbd_prev.ReleaseBuffer();
    m_rgbd_this.ReleaseBuffer();
    m_surfel_motion.ReleaseBuffer();
    releaseTextureCollect(m_opticalflow);
}

void star::OpticalFlowProcessorGMA::ProcessFrame(
    const SurfelMapTex &surfel_map_this,
    const SurfelMapTex &surfel_map_prev,
    const unsigned frame_idx,
    cudaStream_t stream)
{
    if (frame_idx > 0)
    { // OpticalFlow from frame-1
        // Load RBGD
        loadRGBD(surfel_map_this.rgbd, surfel_map_prev.rgbd, stream);

        // Compute opticalflow
        auto img_prev = nn::asTensor(m_rgbd_prev.Ptr(), m_downsample_img_col, m_downsample_img_row);
        auto img_this = nn::asTensor(m_rgbd_this.Ptr(), m_downsample_img_col, m_downsample_img_row);
        auto img_prev_permu = img_prev.permute({2, 0, 1}).unsqueeze(0).to(m_model->GetDevice());
        auto img_this_permu = img_this.permute({2, 0, 1}).unsqueeze(0).to(m_model->GetDevice());
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(img_prev_permu);
        inputs.push_back(img_this_permu);
        try
        {
            m_model->SetInputs(inputs);
            auto time_mark = std::chrono::high_resolution_clock::now();
            m_model->Run();
            auto now = std::chrono::high_resolution_clock::now();
            auto time = now - time_mark;
            std::cout << "GMA runs " << time / std::chrono::milliseconds(1) << " ms" << std::endl;
            // Writing to texture
            if (m_opticalflow_suppress_threshold == 0.f)
            {
                saveOpticalFlow(m_opticalflow, stream);
            }
            else
            {
                saveOpticalFlowWithFilter(m_opticalflow, stream);
            }
        }
        catch (c10::Error &e)
        {
            std::cout << e.what() << std::endl;
        }
        catch (std::exception &e)
        {
            std::cout << e.what() << std::endl;
        }

        // Compute surfel motion
        computeSurfelFlowVisible(surfel_map_prev, surfel_map_this, stream);

        if (m_enable_vis)
        {
            saveContext(frame_idx, stream);
        }
    }
}

void star::OpticalFlowProcessorGMA::coldStart()
{
    for (auto i = 3; i > 0; --i)
    {
        at::Tensor img_0 = torch::ones({1, 4, m_downsample_img_row, m_downsample_img_col}).to(m_model->GetDevice());
        at::Tensor img_1 = torch::ones({1, 4, m_downsample_img_row, m_downsample_img_col}).to(m_model->GetDevice());
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(img_0);
        inputs.push_back(img_1);
        std::cout << "GMA Warming up reverso: " << i << std::endl;
        m_model->SetInputs(inputs);
        // Run multiple times
        try
        {
            m_model->Run();
        }
        catch (std::exception &e)
        {
            std::cout << e.what() << std::endl;
        }
    }
}

void star::OpticalFlowProcessorGMA::loadRGBD(
    const cudaTextureObject_t &rgbd_tex_this,
    const cudaTextureObject_t &rgbd_tex_prev,
    cudaStream_t stream)
{
    if (m_use_static_background)
    {
        loadRGBDWithBackground(rgbd_tex_this, rgbd_tex_prev, stream);
    }
    else
    {
        loadRGBDDirectly(rgbd_tex_this, rgbd_tex_prev, stream);
    }
}

void star::OpticalFlowProcessorGMA::loadRGBDDirectly(
    const cudaTextureObject_t &rgbd_tex_this,
    const cudaTextureObject_t &rgbd_tex_prev,
    cudaStream_t stream)
{

    // This: from measurement
    GArray2D<float4> rgbd_this_2d = GArray2D<float4>(
        m_downsample_img_row,
        m_downsample_img_col,
        m_rgbd_this.Ptr(),
        sizeof(float4) * m_downsample_img_col);
    GArray2D<float4> rgbd_prev_2d = GArray2D<float4>(
        m_downsample_img_row,
        m_downsample_img_col,
        m_rgbd_prev.Ptr(),
        sizeof(float4) * m_downsample_img_col);

    textureToMap2D(rgbd_tex_this, rgbd_this_2d, stream);
    textureToMap2D(rgbd_tex_prev, rgbd_prev_2d, stream);

    // Sync before exit
    cudaSafeCall(cudaStreamSynchronize(stream));
}

void star::OpticalFlowProcessorGMA::loadRGBDWithBackground(
    const cudaTextureObject_t &rgbd_tex_this,
    const cudaTextureObject_t &rgbd_tex_prev,
    cudaStream_t stream)
{
    // Load to this
    unsigned img_cols = m_downsample_img_col;
    unsigned img_rows = m_downsample_img_row;
    dim3 blk(32, 32);
    dim3 grid(divUp(img_cols, blk.x), divUp(img_rows, blk.y));
    // This: from measurement
    device::loadRGBWithBackgroundKernel<<<grid, blk, 0, stream>>>(
        rgbd_tex_this,
        m_rgbd_this.Ptr(),
        m_background_img.ptr(),
        img_cols,
        img_rows);
    // Prev: from previous geometry
    device::loadRGBWithBackgroundKernel<<<grid, blk, 0, stream>>>(
        rgbd_tex_prev,
        m_rgbd_prev.Ptr(),
        m_background_img.ptr(),
        img_cols,
        img_rows);

    // Sync before exit
    cudaSafeCall(cudaStreamSynchronize(stream));
    std::cout << "RGBD loaded with background" << std::endl;
}

void star::OpticalFlowProcessorGMA::saveOpticalFlow(
    CudaTextureSurface &opticalflow_texsurf,
    cudaStream_t stream)
{
    torch::Tensor output_gpu;
    if (m_model->GetDevice() == "cpu")
        output_gpu = m_model->Output().to("cuda:0");
    else
        output_gpu = m_model->Output();

    nn::copyTensorAsync<float2>(
        output_gpu,
        opticalflow_texsurf.d_array,
        m_downsample_img_col,
        m_downsample_img_row,
        stream);
    cudaSafeCall(cudaStreamSynchronize(stream));
}

void star::OpticalFlowProcessorGMA::saveOpticalFlowWithFilter(
    CudaTextureSurface &opticalflow_texsurf,
    cudaStream_t stream)
{
    // 1 - Fetch NN output
    torch::Tensor output_gpu;
    if (m_model->GetDevice() == "cpu")
        output_gpu = m_model->Output().to("cuda:0");
    else
        output_gpu = m_model->Output();

    // 2 - Run opticalflow filter
    unsigned img_cols = m_downsample_img_col;
    unsigned img_rows = m_downsample_img_row;
    dim3 blk(32, 32);
    dim3 grid(divUp(img_cols, blk.x), divUp(img_rows, blk.y));
    device::OpticalFlowFilterKernel<<<grid, blk, 0, stream>>>(
        (float2 *)output_gpu.contiguous().data_ptr(),
        opticalflow_texsurf.surface,
        m_opticalflow_suppress_threshold,
        img_cols,
        img_rows);
    cudaSafeCall(cudaStreamSynchronize(stream));
}

void star::OpticalFlowProcessorGMA::saveContext(const unsigned frame_idx, cudaStream_t stream)
{
    auto &context = easy3d::Context::Instance();
    std::string optical_name = "of";
    // Save images
    context.addImage(optical_name, optical_name);
    visualize::SaveOpticalFlowMap(
        m_opticalflow.texture,
        context.at(optical_name));
}

void star::OpticalFlowProcessorGMA::computeSurfelFlowVisible(
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