#pragma once
#include <cuda_fp16.h>
#include <star/common/ArrayView.h>
#include <star/common/GBufferArray.h>
#include <star/common/common_texture_utils.h>

namespace star
{
    struct Tsdf
    {
        unsigned width;  // cols
        unsigned height; // rows
        unsigned depth;  // height
        float voxel_size;
        float3 origin;
        CudaTextureSurface tsdf_val;    // (float)
        CudaTextureSurface tsdf_weight; // (unchar, 0~255), no need to support more

        void createTsdf(
            const unsigned width_, const unsigned height_, const unsigned depth_,
            const float voxel_size_, const float3 origin_);
        void releaseTsdf();
        void reset(cudaStream_t stream = 0); // Reset to 0
    };

}