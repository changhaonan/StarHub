#include <star/math/device_mat.h>
#include <star/geometry/tsdf/bilinear_interpolation.cuh>
#include <star/geometry/tsdf/tsdf_constants.h>
#include <star/geometry/tsdf/update_tsdf.h>
#include <device_launch_parameters.h>
#include <cuda_fp16.h>

namespace star::device
{
    __device__ __forceinline__ void depth_interpolated_read(
        const cudaTextureObject_t measure_depth_collect,
        const unsigned x, const unsigned y, const unsigned z,
        const float voxel_size, const float3 origin,
        const mat34 world2cam,
        const Intrinsic intrinsic,
        const unsigned img_cols, const unsigned img_rows,
        float &depth_measure,
        float3 &tsdf_center_cam)
    {
        float3 tsdf_center = make_float3(
            float(x) * voxel_size + origin.x,
            float(y) * voxel_size + origin.y,
            float(z) * voxel_size + origin.z);

        tsdf_center_cam = world2cam.rot * tsdf_center + world2cam.trans;
        // Backtrack to pixel
        float pixel_x = (tsdf_center_cam.x / (tsdf_center_cam.z + 1e-6f)) * intrinsic.focal_x + intrinsic.principal_x;
        float pixel_y = (tsdf_center_cam.y / (tsdf_center_cam.z + 1e-6f)) * intrinsic.focal_y + intrinsic.principal_y;

        if (pixel_x < 0.f || pixel_x >= float(img_cols - 1) ||
            pixel_y < 0.f || pixel_y >= float(img_rows - 1))
        { // Edge control, sometimes, texture2D performs strange
            depth_measure = 0.f;
            return;
        }

        // Check the edging status
        unsigned pixel_x_int = __float2uint_rd(pixel_x);
        unsigned pixel_y_int = __float2uint_rd(pixel_y);
        float pixel_x_res = pixel_x - float(pixel_x_int);
        float pixel_y_res = pixel_y - float(pixel_y_int);
        float depth_measure_neighbor[4];
        depth_measure_neighbor[0] = tex2D<float>(measure_depth_collect, pixel_x_int, pixel_y_int);
        depth_measure_neighbor[1] = tex2D<float>(measure_depth_collect, pixel_x_int + 1, pixel_y_int);
        depth_measure_neighbor[2] = tex2D<float>(measure_depth_collect, pixel_x_int + 1, pixel_y_int + 1);
        depth_measure_neighbor[3] = tex2D<float>(measure_depth_collect, pixel_x_int, pixel_y_int + 1);

        // Bilinear interpolation
        if (check_interpolate_smooth(
                depth_measure_neighbor[0], depth_measure_neighbor[1],
                depth_measure_neighbor[2], depth_measure_neighbor[3], d_interpolation_gap))
        {
            depth_measure = bilinear_interpolate(
                pixel_x_res, pixel_y_res,
                depth_measure_neighbor[0], depth_measure_neighbor[1],
                depth_measure_neighbor[2], depth_measure_neighbor[3]);
        }
        else
        {
            depth_measure = 0.f;
        }
    }

    __global__ void UpdateTsdfKernel(
        const cudaTextureObject_t tsdf_val_prev,
        const cudaTextureObject_t tsdf_weight_prev,
        const cudaTextureObject_t measure_depth_collect, // (m)
        cudaSurfaceObject_t tsdf_val_next,
        cudaSurfaceObject_t tsdf_weight_next,
        const unsigned width,
        const unsigned height,
        const unsigned depth,
        const float voxel_size,
        const float3 origin,
        const unsigned img_rows,
        const unsigned img_cols,
        const mat34 world2cam,
        const Intrinsic intrinsic)
    {
        const unsigned x = threadIdx.x + blockDim.x * blockIdx.x;
        const unsigned y = threadIdx.y + blockDim.y * blockIdx.y;
        const unsigned z = threadIdx.z + blockDim.z * blockIdx.z;
        if (x >= width || y >= height || z >= depth)
            return;

        bool update_flag = true;

        // Prev value
        float tsdf_prev = tex3D<float>(tsdf_val_prev, float(x) + 0.5f, float(y) + 0.5f, float(z) + 0.5f);
        unsigned char weight_prev_unchar = tex3D<unsigned char>(tsdf_weight_prev, float(x) + 0.5f, float(y) + 0.5f, float(z) + 0.5f);
        float weight_prev = float(weight_prev_unchar);

        // Measurement
        float depth_measure;
        float3 tsdf_center_cam;
        depth_interpolated_read(
            measure_depth_collect, x, y, z, voxel_size, origin, world2cam, intrinsic, img_cols, img_rows,
            depth_measure, tsdf_center_cam);

        if (fabs(depth_measure) < 1e-6f || fabs(depth_measure) > d_far_clip)
            update_flag = false; // Invalid pixel. Don't update

        /*if (depth_measure < 0.5f && update_flag) {
            printf("x: %f, y: %f, depth: %f.\n", pixel_x, pixel_y, depth_measure);
        }*/
        // Average depth
        weight_prev = min(weight_prev, d_tsdf_weight_max - 1.f);
        float sdf_measure = depth_measure - tsdf_center_cam.z;
        if (sdf_measure < -d_tsdf_threshold)
            update_flag = false;

        if (!update_flag)
        { // Unseen part, keep it the same
            surf3Dwrite(tsdf_prev, tsdf_val_next, sizeof(float) * x, y, z);
            surf3Dwrite((unsigned char)(weight_prev), tsdf_weight_next, sizeof(unsigned char) * x, y, z);
        }
        else
        {
            float tsdf_measure = min(sdf_measure, d_tsdf_threshold) / d_tsdf_threshold;
            float tsdf_update = (tsdf_prev * weight_prev + tsdf_measure) / (weight_prev + 1.f);
            surf3Dwrite(tsdf_update, tsdf_val_next, sizeof(float) * x, y, z);
            surf3Dwrite((unsigned char)(weight_prev + 1.f), tsdf_weight_next, sizeof(unsigned char) * x, y, z);
        }
    }

    __global__ void InspectInterpolationKernel(
        const cudaTextureObject_t measure_depth_collect,
        float4 *__restrict__ interpolated_depth,
        const unsigned width,
        const unsigned height,
        const unsigned depth,
        const float voxel_size,
        const float3 origin,
        const unsigned img_rows,
        const unsigned img_cols,
        const mat34 world2cam,
        const Intrinsic intrinsic,
        unsigned *count)
    {
        const unsigned x = threadIdx.x + blockDim.x * blockIdx.x;
        const unsigned y = threadIdx.y + blockDim.y * blockIdx.y;
        const unsigned z = threadIdx.z + blockDim.z * blockIdx.z;
        if (x >= width || y >= height || z >= depth)
            return;

        float depth_measure;
        float3 tsdf_center_cam;
        depth_interpolated_read(
            measure_depth_collect, x, y, z, voxel_size, origin, world2cam, intrinsic, img_cols, img_rows,
            depth_measure, tsdf_center_cam);

        float sdf_measure = depth_measure - tsdf_center_cam.z;
        if (fabs(sdf_measure) < d_tsdf_threshold)
        {
            tsdf_center_cam.z = depth_measure;
            unsigned old_count = atomicAdd(count, (unsigned)1);
            interpolated_depth[old_count] = make_float4(tsdf_center_cam.x, tsdf_center_cam.y, tsdf_center_cam.z, 1.f);
        }
    }

    __global__ void SectionInspectKernel(
        const cudaTextureObject_t measure_depth_collect, // (m)
        GArraySlice<float4> section_inspect_surfel,
        const unsigned width,
        const unsigned height,
        const unsigned depth,
        const float voxel_size,
        const float3 origin,
        const unsigned img_rows,
        const unsigned img_cols,
        const mat34 world2cam,
        const Intrinsic intrinsic,
        const SIDirection direction,
        const unsigned section_layer,
        unsigned *update_count)
    {
        unsigned x, y, z;
        switch (direction)
        {
        case x_axis:
            x = section_layer;
            y = threadIdx.x + blockDim.x * blockIdx.x;
            z = threadIdx.y + blockDim.y * blockIdx.y;
            break;
        case y_axis:
            x = threadIdx.x + blockDim.x * blockIdx.x;
            y = section_layer;
            z = threadIdx.y + blockDim.y * blockIdx.y;
            break;
        case z_axis:
            x = threadIdx.x + blockDim.x * blockIdx.x;
            y = threadIdx.y + blockDim.y * blockIdx.y;
            z = section_layer;
            break;
        default:
            return;
        }
        if (x >= width || y >= height || z >= depth)
            return;
        bool update_flag = true;

        // Measurement
        float3 tsdf_center = make_float3(
            float(x) * voxel_size + origin.x,
            float(y) * voxel_size + origin.y,
            float(z) * voxel_size + origin.z);

        float3 tsdf_center_cam = world2cam.rot * tsdf_center + world2cam.trans;
        // Backtrack to pixel
        float pixel_x = (tsdf_center_cam.x / (tsdf_center_cam.z + 1e-6f)) * intrinsic.focal_x + intrinsic.principal_x;
        float pixel_y = (tsdf_center_cam.y / (tsdf_center_cam.z + 1e-6f)) * intrinsic.focal_y + intrinsic.principal_y;

        if (pixel_x < 0.f || pixel_x >= float(img_cols - 1) ||
            pixel_y < 0.f || pixel_y >= float(img_rows - 1))
            update_flag = false;
        float depth_measure = tex2D<float>(measure_depth_collect, pixel_x + 0.5f, pixel_y + 0.5f);
        // Reconstruct the surfel
        float4 measure_surfel = make_float4(
            (pixel_x - intrinsic.principal_x) / intrinsic.focal_x * depth_measure,
            (pixel_y - intrinsic.principal_y) / intrinsic.focal_y * depth_measure,
            depth_measure,
            1.f);
        if (!update_flag)
        { // Unseen part, do not use
        }
        else
        {
            unsigned old_count = atomicAdd(update_count, unsigned(1)); // Type is controlled
            section_inspect_surfel[old_count] = make_float4(
                measure_surfel.x,
                measure_surfel.y,
                measure_surfel.z,
                1.f);
        }
    }

}

void star::UpdateTsdf(
    const Tsdf &tsdf_prev,
    Tsdf &tsdf_next,
    const cudaTextureObject_t measure_depth_collect,
    const Intrinsic &intrinsic,
    const Eigen::Matrix4f &cam2world,
    const unsigned img_rows,
    const unsigned img_cols,
    cudaStream_t stream)
{

    dim3 blk(8, 8, 8);
    dim3 grid(divUp(tsdf_prev.width, blk.x), divUp(tsdf_prev.height, blk.y), divUp(tsdf_prev.depth, blk.z));
    device::UpdateTsdfKernel<<<grid, blk, 0, stream>>>(
        tsdf_prev.tsdf_val.texture,
        tsdf_prev.tsdf_weight.texture,
        measure_depth_collect,
        tsdf_next.tsdf_val.surface,
        tsdf_next.tsdf_weight.surface,
        tsdf_prev.width,
        tsdf_prev.height,
        tsdf_prev.depth,
        tsdf_prev.voxel_size,
        tsdf_prev.origin,
        img_rows,
        img_cols,
        // mat34(cam2world),
        mat34(cam2world.inverse()),
        intrinsic);

    // Sync and check error
#if defined(CUDA_DEBUG_SYNC_CHECK)
    cudaSafeCall(cudaStreamSynchronize(stream));
    cudaSafeCall(cudaGetLastError());
#endif
}

void star::InspectInterpolation(
    const Tsdf &tsdf,
    const cudaTextureObject_t measure_depth_collect, // (m) float32
    GBufferArray<float4> &interpolated_depth,
    const Intrinsic &intrinsic,
    const Eigen::Matrix4f &cam2world,
    const unsigned img_rows,
    const unsigned img_cols,
    cudaStream_t stream)
{
    unsigned *count;
    cudaSafeCall(cudaMallocAsync((void **)&count, sizeof(unsigned), stream));
    cudaSafeCall(cudaMemsetAsync(count, 0, sizeof(unsigned), stream));
    dim3 blk(8, 8, 8);
    dim3 grid(divUp(tsdf.width, blk.x), divUp(tsdf.height, blk.y), divUp(tsdf.depth, blk.z));
    device::InspectInterpolationKernel<<<grid, blk, 0, stream>>>(
        measure_depth_collect,
        interpolated_depth.Ptr(),
        tsdf.width,
        tsdf.height,
        tsdf.depth,
        tsdf.voxel_size,
        tsdf.origin,
        img_rows,
        img_cols,
        mat34(cam2world.inverse()),
        intrinsic,
        count);
    unsigned h_count;
    cudaSafeCall(cudaMemcpyAsync(&h_count, count, sizeof(unsigned), cudaMemcpyDeviceToHost));
    cudaSafeCall(cudaStreamSynchronize(stream));
    interpolated_depth.ResizeArrayOrException(h_count);
}

void star::SectionInspectionForUpdate(
    const Tsdf &tsdf,
    const cudaTextureObject_t measure_depth_collect,
    const Intrinsic &intrinsic,
    const Eigen::Matrix4f &cam2world,
    const unsigned img_rows,
    const unsigned img_cols,
    const SIDirection direction,
    const unsigned section_layer,                 // The layer to inspect
    GBufferArray<float4> &section_inspect_surfel, // Measurement inspect that layer
    cudaStream_t stream)
{
    dim3 blk(16, 16);
    dim3 grid;
    switch (direction)
    {
    case x_axis:
        grid = dim3(divUp(tsdf.height, blk.x), divUp(tsdf.depth, blk.y));
        break;
    case y_axis:
        grid = dim3(divUp(tsdf.width, blk.x), divUp(tsdf.depth, blk.y));
        break;
    case z_axis:
        grid = dim3(divUp(tsdf.width, blk.x), divUp(tsdf.height, blk.y));
        break;
    default:
        break;
    }

    unsigned *update_count;
    cudaSafeCall(cudaMallocAsync((void **)&update_count, sizeof(unsigned), stream));
    cudaSafeCall(cudaMemsetAsync(update_count, (unsigned)0, sizeof(unsigned), stream));
    device::SectionInspectKernel<<<grid, blk, 0, stream>>>(
        measure_depth_collect, // (m)
        section_inspect_surfel.Slice(),
        tsdf.width,
        tsdf.height,
        tsdf.depth,
        tsdf.voxel_size,
        tsdf.origin,
        img_rows,
        img_cols,
        mat34(cam2world.inverse()),
        intrinsic,
        direction,
        section_layer,
        update_count);

    // Sync and check error
#if defined(CUDA_DEBUG_SYNC_CHECK)
    cudaSafeCall(cudaStreamSynchronize(stream));
    cudaSafeCall(cudaGetLastError());
#endif

    // Resize
    unsigned h_update_count;
    cudaSafeCall(cudaMemcpyAsync(&h_update_count, update_count, sizeof(unsigned), cudaMemcpyDeviceToHost, stream));
    cudaSafeCall(cudaStreamSynchronize(stream));

    section_inspect_surfel.ResizeArrayOrException(h_update_count);
}