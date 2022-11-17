#include <star/common/common_texture_utils.h>
#include <mono_star/opt/ImageTermKNNFetcher.h>
#include <device_launch_parameters.h>

namespace star::device
{
	// Only mark the pixel that corresponded to valid input
	// Let the term handler to deal with other issues
	__global__ void markPotentialValidImageTermPixelKernel(
		cudaTextureObject_t index_map,
		unsigned img_rows, unsigned img_cols,
		unsigned *__restrict__ reference_pixel_indicator)
	{
		const auto x = threadIdx.x + blockDim.x * blockIdx.x;
		const auto y = threadIdx.y + blockDim.y * blockIdx.y;
		if (x >= img_cols || y >= img_rows)
			return;
		// The indicator will must be written to pixel_occupied_array
		const auto offset = y * img_cols + x;
		// Read the value on index map
		const auto surfel_index = tex2D<unsigned>(index_map, x, y);
		// Need other criterion?
		unsigned indicator = 0;
		if (surfel_index != d_invalid_index)
		{
			indicator = 1;
		}
		reference_pixel_indicator[offset] = indicator;
	}

	__global__ void markPotentialValidImageTermPixelResampleKernel(
		cudaTextureObject_t index_map,
		const float *__restrict__ random_density,
		const float resample_prob,
		unsigned img_rows, unsigned img_cols,
		unsigned *__restrict__ reference_pixel_indicator)
	{
		const auto x = threadIdx.x + blockDim.x * blockIdx.x;
		const auto y = threadIdx.y + blockDim.y * blockIdx.y;
		if (x >= img_cols || y >= img_rows)
			return;
		// The indicator will must be written to pixel_occupied_array
		const auto offset = y * img_cols + x;
		// Read the value on index map
		const auto surfel_index = tex2D<unsigned>(index_map, x, y);
		// Randomly resampled
		unsigned indicator = 0;
		if (surfel_index != d_invalid_index)
		{
			if (resample_prob >= random_density[y * img_cols + x])
				indicator = 1;
		}
		reference_pixel_indicator[offset] = indicator;
	}

	__global__ void compactPontentialImageTermPixelsKernel(
		const GArrayView2D<KNNAndWeight<d_surfel_knn_size>> knn_patch_map,
		cudaTextureObject_t opticalflow_map,
		const unsigned *__restrict__ potential_pixel_indicator,
		const unsigned *__restrict__ prefixsum_pixel_indicator,
		ushort4 *__restrict__ potential_pixel_pair, // (x, y, x + ofx, y + ofy)
		unsigned short *__restrict__ potential_pixels_knn_patch,
		float *__restrict__ potential_pixels_knn_patch_spatial_weight,
		float *__restrict__ potential_pixels_knn_patch_connect_weight)
	{
		const auto x = threadIdx.x + blockDim.x * blockIdx.x;
		const auto y = threadIdx.y + blockDim.y * blockIdx.y;
		if (x >= knn_patch_map.Cols() || y >= knn_patch_map.Rows())
			return;
		const auto flatten_idx = x + y * knn_patch_map.Cols();
		if (potential_pixel_indicator[flatten_idx] > 0)
		{
			const auto offset = prefixsum_pixel_indicator[flatten_idx] - 1;
			const KNNAndWeight<d_surfel_knn_size> knn_patch = knn_patch_map(y, x);
			const float2 opticalflow = tex2D<float2>(opticalflow_map, x, y);
			// const float2 opticalflow = make_float2(0.f, 0.f);
			const unsigned x_ofx = __float2uint_rn(opticalflow.x + float(x));
			const unsigned y_ofy = __float2uint_rn(opticalflow.y + float(y));
			potential_pixel_pair[offset] = make_ushort4(x, y, x_ofx, y_ofy);
#pragma unroll
			for (auto i = 0; i < d_surfel_knn_size; ++i)
			{
				potential_pixels_knn_patch[offset * d_surfel_knn_size + i] = knn_patch.knn[i];
				potential_pixels_knn_patch_spatial_weight[offset * d_surfel_knn_size + i] = knn_patch.spatial_weight[i];
				potential_pixels_knn_patch_connect_weight[offset * d_surfel_knn_size + i] = knn_patch.connect_weight[i];
			}
		}
	}

	__global__ void UpdateKNNTermKernel(
		const DualQuaternion *__restrict__ node_se3,
		const unsigned short *__restrict__ potential_pixels_knn_patch,
		DualQuaternion *__restrict__ potential_pixels_knn_patch_dq,
		const unsigned potential_pixel_size)
	{
		const auto idx = threadIdx.x + blockDim.x * blockIdx.x;
		if (idx >= potential_pixel_size)
			return;
#pragma unroll
		for (auto i = 0; i < d_surfel_knn_size; ++i)
		{
			unsigned nn_idx = potential_pixels_knn_patch[idx * d_surfel_knn_size + i];
			potential_pixels_knn_patch_dq[idx * d_surfel_knn_size + i] = node_se3[nn_idx];
		}
	}
}

star::ImageTermKNNFetcher::ImageTermKNNFetcher() : m_image_height_max(0), m_image_width_max(0)
{
	// The initialization part
	const auto &config = ConfigParser::Instance();
	m_num_cam = config.num_cam();
	for (auto cam_idx = 0; cam_idx < m_num_cam; ++cam_idx)
	{
		m_image_height[cam_idx] = config.downsample_img_rows(cam_idx);
		m_image_width[cam_idx] = config.downsample_img_cols(cam_idx);
		m_image_height_max = (m_image_height_max > m_image_height[cam_idx]) ? m_image_height_max : m_image_height[cam_idx];
		m_image_width_max = (m_image_width_max > m_image_width[cam_idx]) ? m_image_width_max : m_image_width[cam_idx];
	}

	m_resample_prob = config.resample_prob();
	memset(&m_geometry_maps, 0, sizeof(m_geometry_maps));

	// The malloc part
	unsigned num_pixels_all = 0;
	for (auto cam_idx = 0; cam_idx < m_num_cam; ++cam_idx)
	{ // Per-camera
		const auto num_pixels = m_image_height[cam_idx] * m_image_width[cam_idx];
		m_potential_pixel_indicator[cam_idx].create(num_pixels);
		// For compaction
		m_indicator_prefixsum[cam_idx].AllocateBuffer(num_pixels);
		// Fixed
		m_potential_pixels[cam_idx].AllocateBuffer(num_pixels);
		m_dense_image_knn_patch[cam_idx].AllocateBuffer(num_pixels * d_surfel_knn_size);
		m_dense_image_knn_patch_spatial_weight[cam_idx].AllocateBuffer(num_pixels * d_surfel_knn_size);
		m_dense_image_knn_patch_connect_weight[cam_idx].AllocateBuffer(num_pixels * d_surfel_knn_size);
		// Updated
		m_dense_image_knn_patch_dq[cam_idx].AllocateBuffer(num_pixels * d_surfel_knn_size);
		num_pixels_all += num_pixels;
	}

	// Merged buffer
	// Fixed
	m_potential_pixels_all.AllocateBuffer(num_pixels_all);
	m_dense_image_knn_patch_all.AllocateBuffer(num_pixels_all * d_surfel_knn_size);
	m_dense_image_knn_patch_spatial_weight_all.AllocateBuffer(num_pixels_all * d_surfel_knn_size);
	m_dense_image_knn_patch_connect_weight_all.AllocateBuffer(num_pixels_all * d_surfel_knn_size);
	// Updated
	m_dense_image_knn_patch_dq_all.AllocateBuffer(num_pixels_all * d_surfel_knn_size);

	// The page-locked memory
	cudaSafeCall(cudaMallocHost((void **)&m_num_potential_pixel, sizeof(unsigned))); // Page memory: used for cudaMemcpyAsync

	// Used for resampling
	curandCreateGenerator(&m_gen, CURAND_RNG_PSEUDO_DEFAULT);
	curandSetPseudoRandomGeneratorSeed(m_gen, 1234ULL);
}

star::ImageTermKNNFetcher::~ImageTermKNNFetcher()
{
	for (auto i = 0; i < m_num_cam; ++i)
	{ // Per-camera
		m_potential_pixel_indicator[i].release();
		m_potential_pixels[i].ReleaseBuffer();
		m_dense_image_knn_patch[i].ReleaseBuffer();
		m_dense_image_knn_patch_spatial_weight[i].ReleaseBuffer();
		m_dense_image_knn_patch_connect_weight[i].ReleaseBuffer();
		m_dense_image_knn_patch_dq[i].ReleaseBuffer();
	}
	m_potential_pixels_all.ReleaseBuffer();
	m_dense_image_knn_patch_all.ReleaseBuffer();
	m_dense_image_knn_patch_spatial_weight_all.ReleaseBuffer();
	m_dense_image_knn_patch_connect_weight_all.ReleaseBuffer();
	m_dense_image_knn_patch_dq_all.ReleaseBuffer();

	cudaSafeCall(cudaFreeHost(m_num_potential_pixel));
	curandDestroyGenerator(m_gen);
}

void star::ImageTermKNNFetcher::SetInputs(
	const unsigned num_cam,
	const GArrayView2D<KNNAndWeight<d_surfel_knn_size>> *knn_patch_map,
	cudaTextureObject_t *index_map,
	cudaTextureObject_t *opticalflow_map)
{
	for (auto cam_idx = 0; cam_idx < num_cam; ++cam_idx)
	{ // Per-camera
		m_geometry_maps.knn_patch_map[cam_idx] = knn_patch_map[cam_idx];
		m_geometry_maps.index_map[cam_idx] = index_map[cam_idx];
		m_geometry_maps.opticalflow_map[cam_idx] = opticalflow_map[cam_idx];
	}
}

void star::ImageTermKNNFetcher::SetInputs(
	size_t cam_idx,
	const GArrayView2D<KNNAndWeight<d_surfel_knn_size>> knn_patch_map,
	cudaTextureObject_t index_map,
	cudaTextureObject_t opticalflow_map)
{
	m_geometry_maps.knn_patch_map[cam_idx] = knn_patch_map;
	m_geometry_maps.index_map[cam_idx] = index_map;
	m_geometry_maps.opticalflow_map[cam_idx] = opticalflow_map;
}

// Methods for sanity check
void star::ImageTermKNNFetcher::CheckDenseImageTermKNN(size_t cam_idx, const star::GArrayView<unsigned short> dense_image_knn_gpu)
{
	LOG(INFO) << "Check the image term knn against dense depth knn";

	// Should be called after sync
	STAR_CHECK_EQ(m_dense_image_knn_patch[cam_idx].ArraySize(), m_potential_pixels[cam_idx].ArraySize());
	STAR_CHECK_EQ(m_dense_image_knn_patch[cam_idx].ArraySize(), m_dense_image_knn_patch_spatial_weight[cam_idx].ArraySize());
	STAR_CHECK_EQ(m_dense_image_knn_patch[cam_idx].ArraySize(), m_dense_image_knn_patch_connect_weight[cam_idx].ArraySize());
	STAR_CHECK_EQ(m_dense_image_knn_patch[cam_idx].ArraySize(), dense_image_knn_gpu.Size());

	// Download the data
	std::vector<unsigned short> potential_pixel_knn_array, dense_depth_knn_array;
	dense_image_knn_gpu.Download(dense_depth_knn_array);
	m_dense_image_knn_patch[cam_idx].View().Download(potential_pixel_knn_array);

	// Iterates
	//  Do some random check

	// Seems correct
	LOG(INFO) << "Check done! Seems correct!";
}

void star::ImageTermKNNFetcher::FetchKNNTermSync(cudaStream_t stream)
{
	MarkPotentialMatchedPixels(stream);
	CompactPotentialValidPixels(stream);
	SyncQueryCompactedPotentialPixelSize(stream);
	MergeTermKNNFixed(stream);
}

void star::ImageTermKNNFetcher::MarkPotentialMatchedPixels(cudaStream_t stream)
{
	dim3 blk(16, 16);
	dim3 grid;
	for (auto cam_idx = 0; cam_idx < m_num_cam; ++cam_idx)
	{ // Per-camera
		grid.x = divUp(m_image_width[cam_idx], blk.x);
		grid.y = divUp(m_image_height[cam_idx], blk.y);
		device::markPotentialValidImageTermPixelKernel<<<grid, blk, 0, stream>>>(
			m_geometry_maps.index_map[cam_idx],
			m_image_height[cam_idx],
			m_image_width[cam_idx],
			m_potential_pixel_indicator[cam_idx].ptr());

		// Sync and check error
#if defined(CUDA_DEBUG_SYNC_CHECK)
		cudaSafeCall(cudaStreamSynchronize(stream));
		cudaSafeCall(cudaGetLastError());
#endif
	}
}

void star::ImageTermKNNFetcher::MarkPotentialMatchedPixelsResample(
	const GArrayView<float> node_density,
	cudaStream_t stream)
{
	// Generate random density
	float *random_density;
	size_t num_pixel_max = m_image_height_max * m_image_width_max;
	cudaSafeCall(cudaMalloc((void **)&random_density, sizeof(float) * num_pixel_max));

	for (auto cam_idx = 0; cam_idx < m_num_cam; ++cam_idx)
	{ // Per-camera
		const auto num_pixel = m_image_height[cam_idx] * m_image_width[cam_idx];
		// Generate n floats on device
		curandGenerateUniform(m_gen, random_density, num_pixel);

		dim3 blk(16, 16);
		dim3 grid(divUp(m_image_width[cam_idx], blk.x), divUp(m_image_height[cam_idx], blk.y));
		device::markPotentialValidImageTermPixelResampleKernel<<<grid, blk, 0, stream>>>(
			m_geometry_maps.index_map[cam_idx],
			random_density,
			m_resample_prob,
			m_image_height[cam_idx],
			m_image_width[cam_idx],
			m_potential_pixel_indicator[cam_idx].ptr());
	}

	// Sync and check error
#if defined(CUDA_DEBUG_SYNC_CHECK)
	cudaSafeCall(cudaStreamSynchronize(stream));
	cudaSafeCall(cudaGetLastError());
#endif
	cudaSafeCall(cudaFree(random_density));
}

void star::ImageTermKNNFetcher::CompactPotentialValidPixels(cudaStream_t stream)
{
	for (auto cam_idx = 0; cam_idx < m_num_cam; ++cam_idx)
	{
		// Do a prefix sum
		m_indicator_prefixsum[cam_idx].InclusiveSum(m_potential_pixel_indicator[cam_idx], stream);

		// Invoke the kernel
		dim3 blk(16, 16);
		dim3 grid(divUp(m_image_width[cam_idx], blk.x), divUp(m_image_height[cam_idx], blk.y));
		device::compactPontentialImageTermPixelsKernel<<<grid, blk, 0, stream>>>(
			m_geometry_maps.knn_patch_map[cam_idx],
			m_geometry_maps.opticalflow_map[cam_idx],
			m_potential_pixel_indicator[cam_idx],
			m_indicator_prefixsum[cam_idx].valid_prefixsum_array,
			m_potential_pixels[cam_idx].Ptr(),
			m_dense_image_knn_patch[cam_idx].Ptr(),
			m_dense_image_knn_patch_spatial_weight[cam_idx].Ptr(),
			m_dense_image_knn_patch_connect_weight[cam_idx].Ptr());
		// Sync and check error
#if defined(CUDA_DEBUG_SYNC_CHECK)
		cudaSafeCall(cudaStreamSynchronize(stream));
		cudaSafeCall(cudaGetLastError());
#endif
	}
}

void star::ImageTermKNNFetcher::SyncQueryCompactedPotentialPixelSize(cudaStream_t stream)
{
	for (auto cam_idx = 0; cam_idx < m_num_cam; ++cam_idx)
	{ // Per-camera
		// unsigned num_potential_pairs;
		cudaSafeCall(cudaMemcpyAsync(
			m_num_potential_pixel,
			m_indicator_prefixsum[cam_idx].valid_prefixsum_array.ptr() + m_potential_pixel_indicator[cam_idx].size() - 1,
			sizeof(unsigned),
			cudaMemcpyDeviceToHost,
			stream));
		cudaSafeCall(cudaStreamSynchronize(stream));

		m_potential_pixels[cam_idx].ResizeArrayOrException(*m_num_potential_pixel);
		m_dense_image_knn_patch[cam_idx].ResizeArrayOrException(*m_num_potential_pixel * d_surfel_knn_size);
		m_dense_image_knn_patch_spatial_weight[cam_idx].ResizeArrayOrException(*m_num_potential_pixel * d_surfel_knn_size);
		m_dense_image_knn_patch_connect_weight[cam_idx].ResizeArrayOrException(*m_num_potential_pixel * d_surfel_knn_size);
	}
}

void star::ImageTermKNNFetcher::MergeTermKNNFixed(cudaStream_t stream)
{
	unsigned offset = 0;
	for (auto cam_idx = 0; cam_idx < m_num_cam; ++cam_idx)
	{
		const auto num_pixels = m_potential_pixels[cam_idx].ArraySize();
		cudaSafeCall(cudaMemcpyAsync(
			m_potential_pixels_all.Ptr() + offset,
			m_potential_pixels[cam_idx].Ptr(),
			sizeof(ushort4) * num_pixels,
			cudaMemcpyDeviceToDevice,
			stream));
		cudaSafeCall(cudaMemcpyAsync(
			m_dense_image_knn_patch_all.Ptr() + offset * d_surfel_knn_size,
			m_dense_image_knn_patch[cam_idx].Ptr(),
			sizeof(unsigned short) * num_pixels * d_surfel_knn_size,
			cudaMemcpyDeviceToDevice,
			stream));
		cudaSafeCall(cudaMemcpyAsync(
			m_dense_image_knn_patch_spatial_weight_all.Ptr() + offset * d_surfel_knn_size,
			m_dense_image_knn_patch_spatial_weight[cam_idx].Ptr(),
			sizeof(float) * num_pixels * d_surfel_knn_size,
			cudaMemcpyDeviceToDevice,
			stream));
		cudaSafeCall(cudaMemcpyAsync(
			m_dense_image_knn_patch_connect_weight_all.Ptr() + offset * d_surfel_knn_size,
			m_dense_image_knn_patch_connect_weight[cam_idx].Ptr(),
			sizeof(float) * num_pixels * d_surfel_knn_size,
			cudaMemcpyDeviceToDevice,
			stream));
		offset += num_pixels;
	}
	cudaSafeCall(cudaStreamSynchronize(stream));
	m_potential_pixels_all.ResizeArrayOrException(offset);
	m_dense_image_knn_patch_all.ResizeArrayOrException(offset * d_surfel_knn_size);
	m_dense_image_knn_patch_spatial_weight_all.ResizeArrayOrException(offset * d_surfel_knn_size);
	m_dense_image_knn_patch_connect_weight_all.ResizeArrayOrException(offset * d_surfel_knn_size);
}

void star::ImageTermKNNFetcher::MergeTermKNNUpdated(cudaStream_t stream)
{
	unsigned offset = 0;
	for (auto cam_idx = 0; cam_idx < m_num_cam; ++cam_idx)
	{
		const auto num_pixels = m_potential_pixels[cam_idx].ArraySize();
		cudaSafeCall(cudaMemcpyAsync(
			m_dense_image_knn_patch_dq_all.Ptr() + offset * d_surfel_knn_size,
			m_dense_image_knn_patch_dq[cam_idx].Ptr(),
			sizeof(Lie) * num_pixels * d_surfel_knn_size,
			cudaMemcpyDeviceToDevice,
			stream));
		offset += num_pixels;
	}
	cudaSafeCall(cudaStreamSynchronize(stream));
	m_dense_image_knn_patch_dq_all.ResizeArrayOrException(offset * d_surfel_knn_size);
}

void star::ImageTermKNNFetcher::UpdateKnnTermSync(
	const GArrayView<DualQuaternion> node_se3,
	cudaStream_t stream)
{

	dim3 blk(128);
	dim3 grid;
	for (auto cam_idx = 0; cam_idx < m_num_cam; ++cam_idx)
	{
		const auto num_potential_pixel = m_dense_image_knn_patch[cam_idx].ArraySize() / d_surfel_knn_size;
		grid.x = divUp(num_potential_pixel, blk.x);
		device::UpdateKNNTermKernel<<<grid, blk, 0, stream>>>(
			node_se3.Ptr(),
			m_dense_image_knn_patch[cam_idx].Ptr(),
			m_dense_image_knn_patch_dq[cam_idx].Ptr(),
			num_potential_pixel);
		m_dense_image_knn_patch_dq[cam_idx].ResizeArrayOrException(num_potential_pixel);
	}
	cudaSafeCall(cudaStreamSynchronize(stream));
	MergeTermKNNUpdated(stream);

	// Sync and check error
#if defined(CUDA_DEBUG_SYNC_CHECK)
	cudaSafeCall(cudaStreamSynchronize(stream));
	cudaSafeCall(cudaGetLastError());
#endif
}