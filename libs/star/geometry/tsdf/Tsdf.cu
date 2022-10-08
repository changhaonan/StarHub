#include <device_launch_parameters.h>
#include <cuda_fp16.h>
#include <star/geometry/tsdf/Tsdf.h>
#include <star/common/common_texture_utils.h>

namespace star::device
{
    __global__ void ResetTsdfKernel(
        cudaSurfaceObject_t tsdf_val,
        cudaSurfaceObject_t tsdf_weight,
        const unsigned width,
        const unsigned height,
        const unsigned depth)
    {
        const int x = threadIdx.x + blockDim.x * blockIdx.x;
        const int y = threadIdx.y + blockDim.y * blockIdx.y;
        const int z = threadIdx.z + blockDim.z * blockIdx.z;
        if (x >= width || y >= height || z >= depth)
            return;
        surf3Dwrite(0.f, tsdf_val, x * sizeof(float), y, z);
        surf3Dwrite((unsigned char) 0, tsdf_weight, x * sizeof(unsigned char), y, z);
    }
}

void star::Tsdf::createTsdf(
    const unsigned width_, const unsigned height_, const unsigned depth_,
    const float voxel_size_, const float3 origin_)
{
    width = width_;
    height = height_;
    depth = depth_;
    voxel_size = voxel_size_;
    origin = origin_;

    bool interpolationable = true;
    createUchar3DTextureSurface(width, height, depth, tsdf_weight);
    createFloat3DTextureSurface(width, height, depth, tsdf_val, interpolationable);
}

void star::Tsdf::releaseTsdf()
{
    releaseTextureCollect(tsdf_val);
    releaseTextureCollect(tsdf_weight);
}

void star::Tsdf::reset(cudaStream_t stream)
{
    dim3 blk(8, 8, 8);
    dim3 grid(divUp(width, blk.x), divUp(height, blk.y), divUp(depth, blk.z));
    device::ResetTsdfKernel<<<grid, blk, 0, stream>>>(
        tsdf_val.surface,
        tsdf_weight.surface,
        width,
        height,
        depth);

    // Sync and check error
#if defined(CUDA_DEBUG_SYNC_CHECK)
    cudaSafeCall(cudaStreamSynchronize(stream));
    cudaSafeCall(cudaGetLastError());
#endif
}