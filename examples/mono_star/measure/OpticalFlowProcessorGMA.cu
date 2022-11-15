#include <mono_star/measure/OpticalFlowProcessorGMA.h>
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

    // Camera setting
    m_num_cam = config.num_cam();
    STAR_CHECK_EQ(m_num_cam, 1); // Asset mono camera

    m_downsample_img_col = config.downsample_img_cols();
    m_downsample_img_row = config.downsample_img_rows();
    unsigned num_pixel = m_downsample_img_col * m_downsample_img_row;
    // AllocateBuffer
    m_rgbd_prev.AllocateBuffer(num_pixel);
    m_rgbd_this.AllocateBuffer(num_pixel);
    m_rgbd_prev.ResizeArrayOrException(num_pixel);
    m_rgbd_this.ResizeArrayOrException(num_pixel);

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

    // Start with 3 load running if the device is not "cpu"
    if (m_model->GetDevice() != "cpu")
        coldStart();
}

star::OpticalFlowProcessorGMA::~OpticalFlowProcessorGMA()
{
    m_rgbd_prev.ReleaseBuffer();
    m_rgbd_this.ReleaseBuffer();
}

void star::OpticalFlowProcessorGMA::Process(StarStageBuffer &star_stage_buffer_this,
                                            const StarStageBuffer &star_stage_buffer_prev,
                                            cudaStream_t stream,
                                            const unsigned frame_idx)
{
    // Do nothing
    return;
}

void star::OpticalFlowProcessorGMA::ProcessFrame(
    cudaTextureObject_t &rgbd_tex_this,
    cudaTextureObject_t &rgbd_tex_prev,
    CudaTextureSurface &opticalflow_texsurf,
    const unsigned frame_idx,
    cudaStream_t stream)
{
    if (frame_idx > 0)
    { // OpticalFlow from frame-1
        // 1. Load RBGD
        loadRGBD(rgbd_tex_this, rgbd_tex_prev, stream);

        // 2. Run the model
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
                saveOpticalFlow(opticalflow_texsurf, stream);
            }
            else
            {
                saveOpticalFlowWithFilter(opticalflow_texsurf, stream);
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
    cudaTextureObject_t &rgbd_tex_this,
    cudaTextureObject_t &rgbd_tex_prev,
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
    cudaTextureObject_t &rgbd_tex_this,
    cudaTextureObject_t &rgbd_tex_prev,
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
    cudaTextureObject_t &rgbd_tex_this,
    cudaTextureObject_t &rgbd_tex_prev,
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