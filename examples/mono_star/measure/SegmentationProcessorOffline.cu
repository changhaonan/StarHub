#include <mono_star/measure/SegmentationProcessorOffline.h>

namespace star::device
{

    __global__ void SemanticRescaleRemapKernel(
        const int *__restrict__ src,
        const int *__restrict__ remap,
        cudaSurfaceObject_t tar,
        const int scale_width,
        const int scale_height,
        const float scale_inv,
        const int max_num_classes)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int idy = blockIdx.y * blockDim.y + threadIdx.y;
        if (idx >= scale_width || idy >= scale_height)
        {
            return;
        }

        int raw_index = __float2int_rn(float(idy) * scale_inv * scale_width * scale_inv + float(idx) * scale_inv);
        int label = src[raw_index];
        if (label >= 0 && label < max_num_classes)
        {
            surf2Dwrite(remap[label], tar, idx * sizeof(int), idy);
        }
        else
        {
            // Mark as 0 if out of range
            surf2Dwrite(0, tar, idx * sizeof(int), idy);
        }
    }

}

star::SegmentationProcessorOffline::SegmentationProcessorOffline()
{
    auto &config = ConfigParser::Instance();

    // Reader-related
    m_fetcher = std::make_shared<VolumeDeformFileFetch>(config.data_path());
    m_start_frame_idx = config.start_frame_idx();
    m_step_frame = config.step_frame();

    // Semantic priors
    m_semantic_label = config.semantic_label();
    m_max_label = config.max_seg_label();
    prepareReMap();

    // Vis
    m_enable_vis = config.enable_vis();
    m_pcd_size = config.pcd_size();

    // Camera setting
    STAR_CHECK_EQ(config.num_cam(), 1); // Asset mono camera
    m_downsample_img_col = config.downsample_img_cols();
    m_downsample_img_row = config.downsample_img_rows();
    m_cam2world = config.extrinsic()[0];
    m_downsample_scale = config.downsample_scale();

    unsigned num_raw_pixel = config.raw_img_cols() * config.raw_img_rows();
    // AllocateBuffer
    cudaSafeCall(cudaMallocHost(
        (void **)&m_raw_seg_img_buff, num_raw_pixel * sizeof(float2)));
    m_g_raw_seg_img.create(num_raw_pixel);
}

star::SegmentationProcessorOffline::~SegmentationProcessorOffline() {
    cudaSafeCall(cudaFreeHost(m_raw_seg_img_buff));
    m_g_raw_seg_img.release();
}

void star::SegmentationProcessorOffline::prepareReMap()
{
    std::vector<int> h_remap;
    h_remap.resize(m_max_label);
    for (int i = 0; i < m_max_label; ++i)
    {
        h_remap[i] = 0;
    }
    for (auto i = 0; i < m_semantic_label.size(); ++i)
    {
        h_remap[m_semantic_label[i]] = i + 1;
    }
    // All undefined labels are marked as 0
    h_remap[0] = 0;
    m_remap.upload(h_remap);
}

void star::SegmentationProcessorOffline::ProcessFrame(
    const SurfelMap::Ptr& surfel_map,
    const unsigned frame_idx,
    cudaStream_t stream)
{
    // Load segmentation
    loadSegmentation(frame_idx, stream);

    // Save to texture && Remap
    dim3 blk(32, 32);
    dim3 grid(divUp(m_downsample_img_col, blk.x), divUp(m_downsample_img_row, blk.y));
    device::SemanticRescaleRemapKernel<<<grid, blk, 0, stream>>>(
        m_g_raw_seg_img.ptr(),
        m_remap.ptr(),
        surfel_map->Segmentation(),
        m_downsample_img_col,
        m_downsample_img_row,
        1.f / m_downsample_scale,
        m_max_label);
    
    cudaSafeCall(cudaStreamSynchronize(stream));

    // Visualize
    if (m_enable_vis)
    {
        // Bind the texture ref
        m_segmentation_ref = surfel_map->SegmentationReadOnly();
        saveContext(frame_idx, stream);
    }
}

void star::SegmentationProcessorOffline::loadSegmentation(
    const unsigned frame_idx, cudaStream_t stream)
{
    // Load seg image
    auto img_idx = m_start_frame_idx + frame_idx * m_step_frame;
    m_fetcher->FetchSegImage(0, img_idx, m_raw_seg_img);

    // Copy to buffer
    memcpy(m_raw_seg_img_buff, m_raw_seg_img.data, sizeof(int) * m_raw_seg_img.total());
    cudaSafeCall(cudaMemcpyAsync(
        m_g_raw_seg_img.ptr(),
        m_raw_seg_img_buff,
        sizeof(int) * m_raw_seg_img.total(),
        cudaMemcpyHostToDevice,
        stream));
}

void star::SegmentationProcessorOffline::saveContext(const unsigned frame_idx, cudaStream_t stream)
{
    // Save image
    auto &context = easy3d::Context::Instance();
    context.addImage("seg");
    visualize::SaveSemanticMap(m_segmentation_ref, visualize::default_semantic_color_dict, context.at("seg"));
}