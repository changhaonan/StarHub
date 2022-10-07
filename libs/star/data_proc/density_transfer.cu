#include <vector_types.h>
#include <star/common/encode_utils.h>
#include <star/common/color_transfer.h>
#include <star/data_proc/density_transfer.h>
#include <device_launch_parameters.h>

namespace star
{
	namespace device
	{

		__global__ void densityTransferKernel(
			cudaTextureObject_t color_time_map,
			unsigned cols, unsigned rows,
			cudaSurfaceObject_t density_map)
		{
			// Check the position of this kernel
			const auto x = threadIdx.x + blockDim.x * blockIdx.x;
			const auto y = threadIdx.y + blockDim.y * blockIdx.y;
			if (x >= cols || y >= rows)
				return;

			// Take out color_time
			float4 color_time = tex2D<float4>(color_time_map, x, y);

			uchar3 rgb_value;
			float_decode_rgb(color_time.x, rgb_value);
			float3 rgb_value_float = make_float3(rgb_value.x, rgb_value.y, rgb_value.z);
			const float density = rgb2density(rgb_value_float);

			// Write
			// surf2Dwrite<float>(density, density_map, x, y);
			surf2Dwrite(density, density_map, x * sizeof(float), y);
		}
	}
}

void star::density_transfer(
	cudaTextureObject_t color_time_map,
	unsigned rows, unsigned cols,
	cudaSurfaceObject_t density_map,
	cudaStream_t stream)
{
	dim3 blk(16, 16);
	dim3 grid(divUp(cols, blk.x), divUp(rows, blk.y));

	device::densityTransferKernel<<<grid, blk, 0, stream>>>(
		color_time_map,
		cols, rows,
		density_map);

	// Sync and check error
#if defined(CUDA_DEBUG_SYNC_CHECK)
	cudaSafeCall(cudaStreamSynchronize(stream));
	cudaSafeCall(cudaGetLastError());
#endif
}