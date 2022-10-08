#include <star/geometry/geometry_map/SurfelMapInitializer.h>
#include <star/img_proc/surfel_reliable_filter.h>
#include <star/img_proc/generate_maps.h>

namespace star::device
{
    constexpr float d_max_measure_clip = 10.f;

    __global__ void ExtractValidTsdfCenterKernel(
        cudaTextureObject_t tsdf_val,
        float4 *__restrict__ valid_tsdf_center,
        unsigned *__restrict__ valid_count,
        const unsigned width,
        const unsigned height,
        const unsigned depth,
        const float voxel_size,
        const float3 origin,
        const unsigned max_num_surfels)
    {
        const int x = threadIdx.x + blockDim.x * blockIdx.x;
        const int y = threadIdx.y + blockDim.y * blockIdx.y;
        const int z = threadIdx.z + blockDim.z * blockIdx.z;
        if (x >= width || y >= height || z >= depth)
            return;
        float tsdf = tex3D<float>(tsdf_val, float(x) + 0.5f, float(y) + 0.5f, float(z) + 0.5f);
        if (fabs(tsdf) > 1e-6f && fabs(tsdf) < 1.f && (tsdf < 0.f))
        { // Valid part. Not counting 1
            unsigned offset = atomicAdd(valid_count, 1);
            if (offset < max_num_surfels)
            {
                valid_tsdf_center[offset] = make_float4(
                    float(x) * voxel_size + origin.x,
                    float(y) * voxel_size + origin.y,
                    float(z) * voxel_size + origin.z,
                    1.f);
            }
            else
            {
                printf("Offset is %d, exceed %d.\n", offset, max_num_surfels);
            }
        }
    }

    __global__ void ComputeScaledVertexFromDepthKernel(
        const cudaTextureObject_t depth_img_collect, // In (m)
        const cudaTextureObject_t raw_depth_img_collect,
        GArraySlice<float4> measurement_surfel, // In (m)
        const unsigned img_rows,
        const unsigned img_cols,
        const Intrinsic intrinsic)
    {
        const int x = threadIdx.x + blockDim.x * blockIdx.x;
        const int y = threadIdx.y + blockDim.y * blockIdx.y;
        if (x >= img_cols || y >= img_rows)
            return;

        unsigned p_idx = x + y * img_cols;
        float depth_measure = tex2D<float>(depth_img_collect, float(x) + 0.5f, float(y) + 0.5f);
        unsigned short raw_depth_measure = tex2D<unsigned short>(raw_depth_img_collect, float(x), float(y));
        float4 measure_p;
        if (fabs(depth_measure) < 1e-6f || fabs(depth_measure) > d_max_measure_clip)
        { // Upper-bound: 10 m
            measure_p = make_float4(0.f, 0.f, 0.f, 1.f);
        }
        else
        {
            measure_p.x = (float(x) - intrinsic.principal_x) / intrinsic.focal_x * depth_measure;
            measure_p.y = (float(y) - intrinsic.principal_y) / intrinsic.focal_y * depth_measure;
            measure_p.z = depth_measure; // z = 0 will be put to Z-surface
            measure_p.w = 1.f;
        }

        measurement_surfel[p_idx] = measure_p;
    }

    __global__ void ComputeScaledVertexFromDepthKernel(
        const cudaTextureObject_t depth_img_collect,   // In (m)
        cudaSurfaceObject_t measurement_vertex_confid, // In (m)
        const unsigned img_rows,
        const unsigned img_cols,
        const Intrinsic intrinsic,
        const float downsample_scale_inv)
    {
        const int x = threadIdx.x + blockDim.x * blockIdx.x;
        const int y = threadIdx.y + blockDim.y * blockIdx.y;
        if (x >= img_cols || y >= img_rows)
            return;

        float depth_measure = tex2D<float>(depth_img_collect, float(x) * downsample_scale_inv + 0.5f, float(y) * downsample_scale_inv + 0.5f);
        float4 measure_p;
        if (fabs(depth_measure) < 1e-6f || fabs(depth_measure) > d_max_measure_clip)
        { // Upper-bound: 10 m
            measure_p = make_float4(0.f, 0.f, 0.f, 1.f);
        }
        else
        {
            measure_p.x = (float(x) - intrinsic.principal_x) / intrinsic.focal_x * depth_measure;
            measure_p.y = (float(y) - intrinsic.principal_y) / intrinsic.focal_y * depth_measure;
            measure_p.z = depth_measure; // z = 0 will be put to Z-surface
            measure_p.w = 1.f;
        }

        // Write
        surf2Dwrite(measure_p, measurement_vertex_confid, x * sizeof(float4), y);
    }

    __global__ void filterInvalidPixelKernel(
        cudaTextureObject_t depth_map,
        cudaTextureObject_t vertex_confid_map,
        cudaSurfaceObject_t filtered_vertex_confid_map,
        const unsigned img_cols,
        const unsigned img_rows,
        const float clip_near,
        const float clip_far)
    {
        const int x = threadIdx.x + blockDim.x * blockIdx.x;
        const int y = threadIdx.y + blockDim.y * blockIdx.y;
        if (x >= img_cols || y >= img_rows)
            return;

        const auto depth = tex2D<float>(depth_map, x, y); // depth should be in clamp mode
        float4 vertex_confid;
        if (abs(depth) > clip_far || abs(depth) < clip_near)
        { // Invalid pixel
            vertex_confid = make_float4(0.f, 0.f, 0.f, 0.f);
        }
        else
        {
            vertex_confid = tex2D<float4>(vertex_confid_map, x, y);
        }

        // Write: allow for local replacing
        surf2Dwrite(vertex_confid, filtered_vertex_confid_map, x * sizeof(float4), y);
    }

    // Valid is decideded by the vertex confid
    __global__ void ComputeIndexMapKernel(
        cudaTextureObject_t vertex_confid_map,
        cudaSurfaceObject_t index_map,
        unsigned *valid_count,
        const unsigned index_offset,
        const unsigned img_cols,
        const unsigned img_rows)
    {
        const int x = threadIdx.x + blockDim.x * blockIdx.x;
        const int y = threadIdx.y + blockDim.y * blockIdx.y;
        if (x >= img_cols || y >= img_rows)
            return;

        float4 vertex_confid = tex2D<float4>(vertex_confid_map, x, y);
        if (fabs(vertex_confid.z) > 1e-8f)
        {
            unsigned old_count = atomicAdd(valid_count, (unsigned)1);
            surf2Dwrite(old_count + index_offset, index_map, x * sizeof(unsigned), y);
        }
        else
        {
            surf2Dwrite(0xffffffff, index_map, x * sizeof(unsigned), y); // unsigned max
        }
    }
}

star::SurfelMapInitializer::SurfelMapInitializer(
    const unsigned width, const unsigned height,
    const float clip_near, const float clip_far,
    const float surfel_radius_scale, const Intrinsic& intrinsic) :
    m_width(width), m_height(height),
    m_clip_near(clip_near), m_clip_far(clip_far),
    m_surfel_radius_scale(surfel_radius_scale),
    m_intrinsic(intrinsic)
{
    createDepthTextureSurface(height, width, m_raw_depth_img_collect);
    createFloat1TextureSurface(height, width, m_filtered_depth_img_collect);
    createFloat4TextureSurface(height, width, m_raw_vertex_confid);
}

star::SurfelMapInitializer::~SurfelMapInitializer()
{
    releaseTextureCollect(m_raw_depth_img_collect);
    releaseTextureCollect(m_filtered_depth_img_collect);
    releaseTextureCollect(m_raw_vertex_confid);
}

void star::SurfelMapInitializer::UploadDepthImage(
    const GArrayView<unsigned> depth_image,
    cudaStream_t stream)
{
    // 1. Upload depth
    cudaSafeCall(cudaMemcpy2DToArrayAsync(
        m_raw_depth_img_collect.d_array, 
        0, 0, 
        depth_image.Ptr(), 
        m_width * sizeof(unsigned short), 
        m_width * sizeof(unsigned short), 
        m_height, cudaMemcpyDeviceToDevice,
        stream)
    );
    // 2. Filter depth
    filterUnreliableDepth(
        m_raw_depth_img_collect.texture,
        m_filtered_depth_img_collect.surface,
        m_height, m_width,
        m_clip_near, m_clip_far,
        m_intrinsic,
        stream);
}

void star::SurfelMapInitializer::InitFromRGBDImage(
    const GArrayView<uchar3> color_image,
    const GArrayView<unsigned> depth_image,
    const float init_time,
    SurfelMap &surfel_map,
    cudaStream_t stream)
{
    // 0. Checking
    STAR_CHECK_EQ(color_image.Size(), surfel_map.NumPixel());
    STAR_CHECK_EQ(depth_image.Size(), surfel_map.NumPixel());

    // 1. Upload Depth image
    UploadDepthImage(depth_image, stream);

    // 2. Create vertex from depth
    InitFromVertexNormalDepth(surfel_map, stream);

    // 3. Compute color time
    createScaledColorTimeMap(
        color_image,
        m_height,
        m_width,
        1.f, // No-scale here
        init_time,
        surfel_map.m_color_time.surface,
        stream);

    // 4. Compute rgbd image (RGBD map is filtered out, used for optical-flow)
    createScaledRGBDMap( // RGBD image, D is the inverse of depth
        color_image,
        m_filtered_depth_img_collect.texture,
        m_height,
        m_width,
        1.f, // No-scale here
        m_clip_near, m_clip_far,
        surfel_map.m_rgbd.surface,
        stream);
}

void star::SurfelMapInitializer::InitFromVertexNormalDepth(
    SurfelMap &surfel_map,
    cudaStream_t stream)
{
    // 1. Create vertex
    computeScaledVertexFromDepth(
        m_raw_vertex_confid.surface,
        m_intrinsic,
        stream);

    // 2. Filter out invalid pixel
    // For this operation, we allow for in-place texture-surface operation
    filterInvalidPixel(
        m_filtered_depth_img_collect.texture,
        m_raw_vertex_confid.texture,
        surfel_map.m_vertex_confid.surface,
        m_clip_near,
        m_clip_far,
        stream);

    // 3. Compute normal
    createNormalRadiusMap(
        surfel_map.m_vertex_confid.texture,
        m_height,
        m_width,
        surfel_map.m_normal_radius.surface,
        m_surfel_radius_scale,
        stream);
}

void star::SurfelMapInitializer::computeScaledVertexFromDepth(
    cudaSurfaceObject_t vertex_confid_buffer,
    const Intrinsic &intrinsic,
    cudaStream_t stream)
{
    dim3 blk(16, 16);
    dim3 grid(divUp(m_width, blk.x), divUp(m_height, blk.y));
    device::ComputeScaledVertexFromDepthKernel<<<grid, blk, 0, stream>>>(
        m_filtered_depth_img_collect.texture, // In (m)
        vertex_confid_buffer,
        m_height,
        m_width,
        intrinsic,
        1.f); // No re-scale here
}

void star::SurfelMapInitializer::filterInvalidPixel(
    cudaTextureObject_t depth_map,
    cudaTextureObject_t vertex_confid_map,
    cudaSurfaceObject_t filtered_vertex_confid_map,
    const float clip_near,
    const float clip_far,
    cudaStream_t stream)
{
    dim3 blk(16, 16);
    dim3 grid(divUp(m_width, blk.x), divUp(m_height, blk.y));
    device::filterInvalidPixelKernel<<<grid, blk, 0, stream>>>(
        depth_map,
        vertex_confid_map,
        filtered_vertex_confid_map,
        m_width,
        m_height,
        clip_near,
        clip_far);

    // Sync and check error
#if defined(CUDA_DEBUG_SYNC_CHECK)
    cudaSafeCall(cudaStreamSynchronize(stream));
    cudaSafeCall(cudaGetLastError());
#endif
}