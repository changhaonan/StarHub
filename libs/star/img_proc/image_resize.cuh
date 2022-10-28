#pragma once
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace star
{

    namespace device
    {
        // Resize the image to the given size
        template <typename T>
        __global__ void resizeKernel(const T *src, T *dst, int src_width, int src_height, int dst_width, int dst_height)
        {
            int x = blockIdx.x * blockDim.x + threadIdx.x;
            int y = blockIdx.y * blockDim.y + threadIdx.y;

            if (x >= dst_width || y >= dst_height)
                return;

            float x_ratio = (float)src_width / dst_width;
            float y_ratio = (float)src_height / dst_height;

            int x_scale = __float2int_rd(x_ratio * x);
            int y_scale = __float2int_rd(y_ratio * y);

            dst[y * dst_width + x] = src[y_scale * src_width + x_scale];
        }
    }

    template <typename T>
    void ResizeImage(const T *src, T *dst, int src_width, int src_height, int dst_width, int dst_height, cudaStream_t stream)
    {
        dim3 blk(32, 32);
        dim3 grid((dst_width + blk.x - 1) / blk.x, (dst_height + blk.y - 1) / blk.y);

        device::resizeKernel<T><<<grid, blk, 0, stream>>>(src, dst, src_width, src_height, dst_width, dst_height);
    }

}