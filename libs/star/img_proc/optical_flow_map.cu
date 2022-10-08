#include <device_launch_parameters.h>
#include <star/img_proc/generate_maps.h>

namespace star::device
{
    __global__ void createScaledOpticalFlowMapKernel(
        const float2 *__restrict__ raw_opticalflow_map,
        const unsigned raw_rows, const unsigned raw_cols,
        const unsigned scaled_rows, const unsigned scaled_cols,
        const float window_size, // Average by window size
        const float scale,
        cudaSurfaceObject_t opticalflow_map)
    {
        const auto x = threadIdx.x + blockIdx.x * blockDim.x;
        const auto y = threadIdx.y + blockIdx.y * blockDim.y;
        if (x >= scaled_cols || y >= scaled_rows)
            return;

        // From here, the access to raw_rgb_img should be in range
        // corresponding pixel in the original image
        // TODO: currently, we are only doing sampling, change it to average if need?
        const auto raw_x = __float2int_rn(x * window_size);
        const auto raw_y = __float2int_rn(y * window_size);
        const auto raw_flatten = raw_x + raw_y * raw_cols;
        const float2 raw_opticalflow = raw_opticalflow_map[raw_flatten];
        const float2 scaled_opticalflow = make_float2(raw_opticalflow.x * scale, raw_opticalflow.y * scale);

        // Construct the result and store it
        surf2Dwrite(scaled_opticalflow, opticalflow_map, x * sizeof(float2), y);
    }
}

void star::createScaledOpticalFlowMap( // Optical flow map creation, but scaled
    const float2 *__restrict__ raw_opticalflow_map,
    const unsigned raw_rows, const unsigned raw_cols,
    const float scale,
    cudaSurfaceObject_t opticalflow_map,
    cudaStream_t stream)
{
    float window_size = 1.f / scale; // Scale should be smaller than 1, window size is larger than one
    unsigned scaled_cols = std::floor(float(raw_cols) * scale);
    unsigned scaled_rows = std::floor(float(raw_rows) * scale);

    dim3 blk(16, 16);
    dim3 grid(divUp(scaled_cols, blk.x), divUp(scaled_rows, blk.y));
    device::createScaledOpticalFlowMapKernel<<<grid, blk, 0, stream>>>(
        raw_opticalflow_map,
        raw_rows, raw_cols,
        scaled_rows, scaled_cols,
        window_size, // Average by window size
        scale,
        opticalflow_map);
    // Sync and check error
#if defined(CUDA_DEBUG_SYNC_CHECK)
    cudaSafeCall(cudaStreamSynchronize(stream));
    cudaSafeCall(cudaGetLastError());
#endif
}
