#include <star/common/common_types.h>
#include <star/common/encode_utils.h>
#include <star/common/color_transfer.h>
#include <star/common/common_texture_utils.h>
#include <star/data_proc/generate_maps.h>
#include <star/data_proc/rgb_clip_normalize.h>
#include <device_launch_parameters.h>

// Local macro
#define boundary_clip 20

namespace star::device
{
	__global__ void clipNormalizeRGBImageKernel(
		const PtrSz<const uchar3> raw_rgb_img,
		const unsigned clip_rows, const unsigned clip_cols,
		cudaSurfaceObject_t clip_rgb_img)
	{
		// Check the position of this kernel
		const auto clip_x = threadIdx.x + blockDim.x * blockIdx.x;
		const auto clip_y = threadIdx.y + blockDim.y * blockIdx.y;
		if (clip_x >= clip_cols || clip_y >= clip_rows)
			return;

		// From here, the access to raw_rgb_img should be in range
		const auto raw_x = clip_x + boundary_clip;
		const auto raw_y = clip_y + boundary_clip;
		const auto raw_flatten = raw_x + raw_y * (clip_cols + 2 * boundary_clip);
		const uchar3 raw_pixel = raw_rgb_img[raw_flatten];

		// Normalize and write to output
		float4 noramlized_rgb;
		noramlized_rgb.x = float(raw_pixel.x) / 255.0f;
		noramlized_rgb.y = float(raw_pixel.y) / 255.0f;
		noramlized_rgb.z = float(raw_pixel.z) / 255.0f;
		noramlized_rgb.w = 1.0f;
		surf2Dwrite(noramlized_rgb, clip_rgb_img, clip_x * sizeof(float4), clip_y);
	}

	__global__ void clipNormalizeRGBImageKernel(
		const PtrSz<const uchar3> raw_rgb_img,
		const unsigned clip_rows, const unsigned clip_cols,
		cudaSurfaceObject_t clip_rgb_img,
		cudaSurfaceObject_t density_map)
	{
		// Check the position of this kernel
		const auto clip_x = threadIdx.x + blockDim.x * blockIdx.x;
		const auto clip_y = threadIdx.y + blockDim.y * blockIdx.y;
		if (clip_x >= clip_cols || clip_y >= clip_rows)
			return;

		// From here, the access to raw_rgb_img should be in range
		const auto raw_x = clip_x + boundary_clip;
		const auto raw_y = clip_y + boundary_clip;
		const auto raw_flatten = raw_x + raw_y * (clip_cols + 2 * boundary_clip);
		const uchar3 raw_pixel = raw_rgb_img[raw_flatten];

		// Normalize and write to output
		float4 noramlized_rgb;
		noramlized_rgb.x = float(raw_pixel.x) / 255.0f;
		noramlized_rgb.y = float(raw_pixel.y) / 255.0f;
		noramlized_rgb.z = float(raw_pixel.z) / 255.0f;
		noramlized_rgb.w = 1.0f;
		const float density = rgba2density(noramlized_rgb); // TODO:Density function can be changed here

		surf2Dwrite(noramlized_rgb, clip_rgb_img, clip_x * sizeof(float4), clip_y);
		surf2Dwrite(density, density_map, clip_x * sizeof(float), clip_y);
	}

	__global__ void createColorTimeMapKernel(
		const PtrSz<const uchar3> raw_rgb_img,
		const unsigned clip_rows, const unsigned clip_cols,
		const float init_time,
		cudaSurfaceObject_t color_time_map)
	{
		const auto clip_x = threadIdx.x + blockIdx.x * blockDim.x;
		const auto clip_y = threadIdx.y + blockIdx.y * blockDim.y;
		if (clip_x >= clip_cols || clip_y >= clip_rows)
			return;

		// From here, the access to raw_rgb_img should be in range
		const auto raw_x = clip_x + boundary_clip;
		const auto raw_y = clip_y + boundary_clip;
		const auto raw_flatten = raw_x + raw_y * (clip_cols + 2 * boundary_clip);
		const uchar3 raw_pixel = raw_rgb_img[raw_flatten];
		const float encoded_pixel = float_encode_rgb(raw_pixel);

		// Construct the result and store it
		const float4 color_time_value = make_float4(encoded_pixel, 0, init_time, init_time);
		surf2Dwrite(color_time_value, color_time_map, clip_x * sizeof(float4), clip_y);
	}

	__global__ void createScaledColorTimeMapKernel(
		const PtrSz<const uchar3> raw_rgb_img,
		const unsigned raw_rows, const unsigned raw_cols,
		const unsigned scaled_rows, const unsigned scaled_cols,
		const float window_size, // Average by window size
		const float init_time,
		cudaSurfaceObject_t color_time_map)
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
		const uchar3 raw_pixel = raw_rgb_img[raw_flatten];
		const float encoded_pixel = float_encode_rgb(raw_pixel);

		// Construct the result and store it
		const float4 color_time_value = make_float4(encoded_pixel, 0, init_time, init_time);
		surf2Dwrite(color_time_value, color_time_map, x * sizeof(float4), y);
	}

	__global__ void createScaledRGBDMapKernel(
		const PtrSz<const uchar3> raw_rgb_img,
		cudaTextureObject_t filtered_depth_img,
		const unsigned raw_rows, const unsigned raw_cols,
		const unsigned scaled_rows, const unsigned scaled_cols,
		const float window_size, // Average by window size
		const float clip_near, const float clip_far,
		cudaSurfaceObject_t rgbd_map)
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
		const uchar3 raw_pixel = raw_rgb_img[raw_flatten];
		const float depth_val = tex2D<float>(filtered_depth_img, raw_x, raw_y);

		float4 color_depth;
		if (fabs(depth_val) <= clip_near || depth_val >= clip_far)
		{
			color_depth = make_float4(0.f, 0.f, 0.f, 0.f);
		}
		else
		{
			// Construct the result and store it
			color_depth = make_float4(
				float(raw_pixel.x) / 127.5f - 1.0f,
				float(raw_pixel.y) / 127.5f - 1.0f,
				float(raw_pixel.z) / 127.5f - 1.0f,
				1.f / depth_val);
		}
		surf2Dwrite(color_depth, rgbd_map, x * sizeof(float4), y);
	}

	__global__ void filterDensityMapKernel(
		cudaTextureObject_t density_map,
		unsigned rows, unsigned cols,
		cudaSurfaceObject_t filter_density_map)
	{
		const auto x = threadIdx.x + blockIdx.x * blockDim.x;
		const auto y = threadIdx.y + blockIdx.y * blockDim.y;
		if (x >= cols || y >= rows)
			return;

		const auto half_width = 5;
		const float center_density = tex2D<float>(density_map, x, y);

		// The window search: Gaussian average
		float sum_all = 0.0f;
		float sum_weight = 0.0f;
		for (auto y_idx = y - half_width; y_idx <= y + half_width; y_idx++)
		{
			for (auto x_idx = x - half_width; x_idx <= x + half_width; x_idx++)
			{
				const float density = tex2D<float>(density_map, x_idx, y_idx);
				const float value_diff2 = (center_density - density) * (center_density - density);
				const float pixel_diff2 = (x_idx - x) * (x_idx - x) + (y_idx - y) * (y_idx - y);
				const float this_weight = (density > 0.0f) * expf(-(1.0f / 25) * pixel_diff2) * expf(-(1.0f / 0.01) * value_diff2);
				sum_weight += this_weight;
				sum_all += this_weight * density;
			}
		}

		// The filter value
		float filter_density_value = sum_all / (sum_weight);

		// Clip the value to suitable range
		if (filter_density_value >= 1.0f)
		{
			filter_density_value = 1.0f;
		}
		else if (filter_density_value >= 0.0f)
		{
			// pass
		}
		else
		{
			filter_density_value = 0.0f;
		}
		// if(isnan(filter_density_value)) printf("Nan in the image");
		surf2Dwrite(filter_density_value, filter_density_map, x * sizeof(float), y);
	}

	__global__ void createNormalizedRGBMapKernel(
		const PtrSz<const uchar3> raw_rgb_img,
		cudaSurfaceObject_t normalized_rgb_map,
		const unsigned rows, const unsigned cols)
	{
		const auto x = threadIdx.x + blockIdx.x * blockDim.x;
		const auto y = threadIdx.y + blockIdx.y * blockDim.y;
		if (x >= cols || y >= rows)
			return;
		const auto raw_flatten = x + y * cols;
		const uchar3 raw_pixel = raw_rgb_img[raw_flatten];

		// Construct the result and store it
		// FIXME: float3 seems not working, so use float4 through padding
		float4 normalized_color = make_float4(
			float(raw_pixel.x) / 255.f,
			float(raw_pixel.y) / 255.f,
			float(raw_pixel.z) / 255.f,
			0.f);

		surf2Dwrite(normalized_color, normalized_rgb_map, x * sizeof(float4), y);
	}

	__global__ void createScaledDepthMapKernel(
		cudaTextureObject_t filtered_depth_img,
		const unsigned scaled_rows, const unsigned scaled_cols,
		const float window_size, // Average by window size
		cudaSurfaceObject_t scaled_depth_map)
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
		const float depth_val = tex2D<float>(filtered_depth_img, raw_x, raw_y);
		surf2Dwrite(depth_val, scaled_depth_map, x * sizeof(float), y);
	}
};

void star::clipNormalizeRGBImage(
	const GArray<uchar3> raw_rgb_img,
	const unsigned clip_rows, const unsigned clip_cols,
	cudaSurfaceObject_t clip_rgb_img,
	cudaStream_t stream)
{
	dim3 blk(16, 16);
	dim3 grid(divUp(clip_cols, blk.x), divUp(clip_rows, blk.y));
	device::clipNormalizeRGBImageKernel<<<grid, blk, 0, stream>>>(
		raw_rgb_img,
		clip_rows, clip_cols,
		clip_rgb_img);

	// Sync and check error
#if defined(CUDA_DEBUG_SYNC_CHECK)
	cudaSafeCall(cudaStreamSynchronize(stream));
	cudaSafeCall(cudaGetLastError());
#endif
}

void star::clipNormalizeRGBImage(
	const GArray<uchar3> &raw_rgb_img,
	unsigned clip_rows, unsigned clip_cols,
	cudaSurfaceObject_t clip_rgb_img,
	cudaSurfaceObject_t density_map,
	cudaStream_t stream)
{
	dim3 blk(16, 16);
	dim3 grid(divUp(clip_cols, blk.x), divUp(clip_rows, blk.y));
	device::clipNormalizeRGBImageKernel<<<grid, blk, 0, stream>>>(
		raw_rgb_img,
		clip_rows, clip_cols,
		clip_rgb_img,
		density_map);

	// Sync and check error
#if defined(CUDA_DEBUG_SYNC_CHECK)
	cudaSafeCall(cudaStreamSynchronize(stream));
	cudaSafeCall(cudaGetLastError());
#endif
}

void star::createColorTimeMap(
	const GArray<uchar3> raw_rgb_img,
	const unsigned clip_rows, const unsigned clip_cols,
	const float init_time,
	cudaSurfaceObject_t color_time_map,
	cudaStream_t stream)
{
	dim3 blk(16, 16);
	dim3 grid(divUp(clip_cols, blk.x), divUp(clip_rows, blk.y));
	device::createColorTimeMapKernel<<<grid, blk, 0, stream>>>(
		raw_rgb_img,
		clip_rows, clip_cols,
		init_time,
		color_time_map);

	// Sync and check error
#if defined(CUDA_DEBUG_SYNC_CHECK)
	cudaSafeCall(cudaStreamSynchronize(stream));
	cudaSafeCall(cudaGetLastError());
#endif
}

void star::createScaledColorTimeMap( // Color time map creation, but scaled
	const GArray<uchar3> raw_rgb_img,
	const unsigned raw_rows, const unsigned raw_cols,
	const float scale,
	const float init_time,
	cudaSurfaceObject_t color_time_map,
	cudaStream_t stream)
{
	float window_size = 1.f / scale; // Scale should be smaller than 1, window size is larger than one
	unsigned scaled_cols = std::floor(float(raw_cols) * scale);
	unsigned scaled_rows = std::floor(float(raw_rows) * scale);

	dim3 blk(16, 16);
	dim3 grid(divUp(scaled_cols, blk.x), divUp(scaled_rows, blk.y));
	device::createScaledColorTimeMapKernel<<<grid, blk, 0, stream>>>(
		raw_rgb_img,
		raw_rows, raw_cols,
		scaled_rows, scaled_cols,
		window_size, // Average by window size
		init_time,
		color_time_map);

	// Sync and check error
#if defined(CUDA_DEBUG_SYNC_CHECK)
	cudaSafeCall(cudaStreamSynchronize(stream));
	cudaSafeCall(cudaGetLastError());
#endif
}

void star::createScaledRGBDMap( // RGBD image, D is the inverse of depth
	const GArray<uchar3> raw_rgb_img,
	cudaTextureObject_t filtered_depth_img,
	const unsigned raw_rows, const unsigned raw_cols,
	const float scale,
	const float clip_near, const float clip_far,
	cudaSurfaceObject_t rgbd_map,
	cudaStream_t stream)
{
	float window_size = 1.f / scale; // Scale should be smaller than 1, window size is larger than one
	unsigned scaled_cols = std::floor(float(raw_cols) * scale);
	unsigned scaled_rows = std::floor(float(raw_rows) * scale);

	dim3 blk(16, 16);
	dim3 grid(divUp(scaled_cols, blk.x), divUp(scaled_rows, blk.y));
	device::createScaledRGBDMapKernel<<<grid, blk, 0, stream>>>(
		raw_rgb_img,
		filtered_depth_img,
		raw_rows, raw_cols,
		scaled_rows, scaled_cols,
		window_size, // Average by window size
		clip_near, clip_far,
		rgbd_map);

	// Sync and check error
#if defined(CUDA_DEBUG_SYNC_CHECK)
	cudaSafeCall(cudaStreamSynchronize(stream));
	cudaSafeCall(cudaGetLastError());
#endif
}

void star::createNormalizedRGBMap(
	const GArray<uchar3> raw_rgb_img,
	cudaSurfaceObject_t normalized_rgb_map,
	const unsigned rows, const unsigned cols,
	cudaStream_t stream)
{
	dim3 blk(16, 16);
	dim3 grid(divUp(cols, blk.x), divUp(rows, blk.y));
	device::createNormalizedRGBMapKernel<<<grid, blk, 0, stream>>>(
		raw_rgb_img, normalized_rgb_map,
		rows, cols);
}

void star::filterDensityMap(
	cudaTextureObject_t density_map,
	cudaSurfaceObject_t filter_density_map,
	unsigned rows, unsigned cols,
	cudaStream_t stream)
{
	dim3 blk(16, 16);
	dim3 grid(divUp(cols, blk.x), divUp(rows, blk.y));
	device::filterDensityMapKernel<<<grid, blk, 0, stream>>>(
		density_map,
		rows, cols,
		filter_density_map);

	// Sync and check error
#if defined(CUDA_DEBUG_SYNC_CHECK)
	cudaSafeCall(cudaStreamSynchronize(stream));
	cudaSafeCall(cudaGetLastError());
#endif
}

void star::createScaledDepthMap(
	cudaTextureObject_t filtered_depth_img,
	const unsigned raw_rows, const unsigned raw_cols,
	const float scale,
	cudaSurfaceObject_t scaled_depth_map,
	cudaStream_t stream)
{
	float window_size = 1.f / scale; // Scale should be smaller than 1, window size is larger than one
	unsigned scaled_cols = std::floor(float(raw_cols) * scale);
	unsigned scaled_rows = std::floor(float(raw_rows) * scale);

	dim3 blk(16, 16);
	dim3 grid(divUp(scaled_cols, blk.x), divUp(scaled_rows, blk.y));
	device::createScaledDepthMapKernel<<<grid, blk, 0, stream>>>(
		filtered_depth_img,
		scaled_rows, scaled_cols,
		window_size, // Average by window size
		scaled_depth_map);
}