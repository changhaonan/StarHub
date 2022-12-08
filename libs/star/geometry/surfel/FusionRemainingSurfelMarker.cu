#include <star/common/common_texture_utils.h>
#include <star/common/surfel_types.h>
#include <star/math/vector_ops.hpp>
#include <device_launch_parameters.h>
#include "FusionRemainingSurfelMarker.h"

namespace star::device
{

	constexpr float d_diff_inlier_thresh = 0.005f;	 // Within which is inlier, 5mm
	constexpr float d_diff_miss_thresh = 0.06f;		 // Outof which is missed match
	constexpr float d_differ_outlier_thresh = 0.03f; // Outlier if above measure by 3cm

	__device__ __forceinline__ float truncated_diff_z(
		const float geometry_vertex_camera_z,
		const float data_vertex_camera_z,
		const float truncated_threshold)
	{
		float z_diff = geometry_vertex_camera_z - data_vertex_camera_z;
		// if (z_diff > truncated_threshold) z_diff = 0;  // If large too much, then it is no match
		if (z_diff > 0.f)
			z_diff = 0; // Only allow for one-direction error
		return fabs(z_diff);
	}

	struct RemainingSurfelMarkerDevice
	{
		// Some constants defined as enum
		enum
		{
			scale_factor = d_fusion_map_scale,
			observation_window_halfsize = 1, // Window size defines the removal sensity
			geometry_window_halfsize = scale_factor,
		};

		// The observation map
		struct
		{
			cudaTextureObject_t vertex_confid_map;
			cudaTextureObject_t normal_radius_map;
			cudaTextureObject_t index_map;
			cudaTextureObject_t color_time_map;
		} observation_maps;

		// The geometry model input
		struct
		{
			cudaTextureObject_t vertex_confid_map;
			cudaTextureObject_t normal_radius_map;
			cudaTextureObject_t color_time_map;
			cudaTextureObject_t index_map;
		} fusion_maps;

		// The remainin surfel indicator
		unsigned *remaining_surfel;
		// The alignement showing: Used for debug
		float *remaining_alignment_error;

		// the camera and time information
		mat34 world2camera;
		float current_time;

		// The global information
		Intrinsic intrinsic;
		unsigned img_cols, img_rows;

		/*
		 * \brief This function do the following things:
		 * 1. Compute the alignment error across different frames.
		 * 2. Mark those overlapped & not valid surfel as not remaining.
		 */
		__device__ __forceinline__ void preProcessMarking() const
		{
			const int x = threadIdx.x + blockDim.x * blockIdx.x;
			const int y = threadIdx.y + blockDim.y * blockIdx.y;

			if (x < geometry_window_halfsize || x >= img_cols - geometry_window_halfsize ||
				y < geometry_window_halfsize || y >= img_rows - geometry_window_halfsize)
			{
				return;
			}

			const float4 vertex_confid_world = tex2D<float4>(fusion_maps.vertex_confid_map, x, y);
			const float4 normal_radius_world = tex2D<float4>(fusion_maps.normal_radius_map, x, y);
			// Apply inverse transform
			const float3 vertex_cam_v3 = world2camera.apply_se3(vertex_confid_world);
			const float3 normal_cam_v3 = world2camera.rot * normal_radius_world;

			const float4 color_time = tex2D<float4>(fusion_maps.color_time_map, x, y);
			const unsigned geometry_index = tex2D<unsigned>(fusion_maps.index_map, x, y);
			if (geometry_index == 0xFFFFFFFF)
				return; // Not valid

			// Check if normal, camera consistency
			const float3 vertex = make_float3(vertex_cam_v3.x, vertex_cam_v3.y, vertex_cam_v3.z);
			const auto array_len_cam = norm(vertex);
			float3 array_dir_cam = -vertex * (1.f / array_len_cam);
			// if (fabs(dot(normal_cam_v3, array_dir_cam)) < 0.2f) return;  // Only consider those surfels in the front, side part doesn't deal with it.

			// Window search
			// Free space violation
			const int ob_x = __float2int_rn(float(x) / float(scale_factor));
			const int ob_y = __float2int_rn(float(y) / float(scale_factor));
			bool depth_matched = false;
			float alignment_error = 1e6f;
			for (auto map_x = ob_x - observation_window_halfsize; map_x < ob_x + observation_window_halfsize; map_x++)
			{
				for (auto map_y = ob_y - observation_window_halfsize; map_y < ob_y + observation_window_halfsize; map_y++)
				{
					const float4 map_vertex_confid = tex2D<float4>(observation_maps.vertex_confid_map, map_x, map_y);
					const float4 map_normal_radius = tex2D<float4>(observation_maps.normal_radius_map, map_x, map_y);

					// if (!is_zero_vertex(map_vertex_confid) && dotxyz(normal_cam_v3, map_normal_radius) > 0.0f) {
					if (!is_zero_vertex(map_vertex_confid))
					{
						depth_matched = true;
						const auto error = truncated_diff_z(vertex.z, map_vertex_confid.z, d_diff_inlier_thresh);
						if (error < alignment_error)
							alignment_error = error;
					}
				}
			}

			// Alignment from all perspective
			if (depth_matched && (alignment_error < remaining_alignment_error[geometry_index] ||
								  remaining_alignment_error[geometry_index] == d_diff_inlier_thresh / 2.f))
			{
				remaining_alignment_error[geometry_index] = alignment_error; // Update it with the smaller error
			}

			// Self-overlap check
			bool overlap_flag = false;
			for (auto neighbor_x = x - geometry_window_halfsize; neighbor_x < x + geometry_window_halfsize; neighbor_x++)
			{
				for (auto neighbor_y = y - geometry_window_halfsize; neighbor_y < y + geometry_window_halfsize; neighbor_y++)
				{
					if (neighbor_x == x && neighbor_y == y)
						continue;
					const auto neighbor_index = tex2D<unsigned>(fusion_maps.index_map, neighbor_x, neighbor_y);
					const float4 neighbor_vertex_confid_world = tex2D<float4>(fusion_maps.vertex_confid_map, neighbor_x, neighbor_y);
					const float4 neighbor_normal_radius_world = tex2D<float4>(fusion_maps.normal_radius_map, neighbor_x, neighbor_y);
					const float3 neighbor_vertex_cam_v3 = world2camera.apply_se3(neighbor_vertex_confid_world);
					const float3 neighbor_normal_cam_v3 = world2camera.rot * neighbor_normal_radius_world;
					if (neighbor_index == 0xFFFFFFFF)
						continue;

					// Check overlap
					const float dist_square = squared_distance(neighbor_vertex_cam_v3, vertex_cam_v3);
					const float dot_value = dot(neighbor_normal_cam_v3, normal_cam_v3);

					if (dot_value > 0.8f && dist_square < 0.0005f * 0.0005f)
					{ // 0.5mm
						// Mark the one with lower confidence or lower index
						if (neighbor_vertex_confid_world.w > vertex_confid_world.w)
						{
							overlap_flag = true;
							break;
						}
						else if (neighbor_vertex_confid_world.w == vertex_confid_world.w)
						{
							if (neighbor_index < geometry_index)
							{
								overlap_flag = true;
								break;
							}
						}
					}
				}
			}

			// Unstable check
			bool just_updated = true;
			if (current_time - last_observed_time(color_time) > 10.f)
				just_updated = false;

			bool surfel_confident = true;
			if (vertex_confid_world.w < 10.f)
				surfel_confident = false;

			// Remaining deciding
			unsigned keep_indicator = 1;
			if ((!just_updated && !surfel_confident) || overlap_flag)
				keep_indicator = 0;
			// if (outlier_flag) keep_indicator = 0;

			// Write to output
			unsigned remaining_status = remaining_surfel[geometry_index];
			if (keep_indicator == 0 && remaining_surfel[geometry_index] == 1)
			{
				remaining_surfel[geometry_index] = 0;
			}
		}
	};

	__global__ void updateRemainingSurfelKernel(
		const RemainingSurfelMarkerDevice marker)
	{
		marker.preProcessMarking();
	}

	__global__ void PostProcessMarking(
		const float *remaining_alignment_error,
		unsigned *remaining_surfel,
		const unsigned num_valid_surfel)
	{
		const int idx = threadIdx.x + blockDim.x * blockIdx.x;
		if (idx >= num_valid_surfel)
			return;

		if (remaining_alignment_error[idx] > d_diff_inlier_thresh)
		{
			remaining_surfel[idx] = 0;
		}
	}

	__global__ void InitializationKernel(
		unsigned *indicator,
		float *error,
		const unsigned array_size)
	{
		const auto idx = threadIdx.x + blockDim.x * blockIdx.x;
		if (idx >= array_size)
			return;
		indicator[idx] = 1;
		error[idx] = d_diff_inlier_thresh / 2.f;
	}

	/* Surfel-warp version operation
	 */
	struct RemainingSurfelMarkerDeviceSurfelWarp
	{
		// Some constants defined as enum
		enum
		{
			scale_factor = d_fusion_map_scale,
			window_halfsize = scale_factor * 2,
			front_threshold = scale_factor * scale_factor * 3,
		};

		// The rendered fusion maps
		struct
		{
			cudaTextureObject_t vertex_confid_map;
			cudaTextureObject_t normal_radius_map;
			cudaTextureObject_t index_map;
			cudaTextureObject_t color_time_map;
		} fusion_maps;

		// The geometry model input
		struct
		{
			GArrayView<float4> vertex_confid;
			const float4 *normal_radius;
			const float4 *color_time;
		} live_geometry;

		// The measure map
		struct
		{
			cudaTextureObject_t depth4removal_map;
		} measure_maps;

		// The remainin surfel indicator from the fuser
		mutable unsigned *remaining_surfel;

		// the camera and time information
		mat34 world2camera;
		float current_time;

		// The global information
		Intrinsic intrinsic;

		__device__ __forceinline__ void processMarking() const
		{
			const auto idx = threadIdx.x + blockDim.x * blockIdx.x;
			if (idx >= live_geometry.vertex_confid.Size())
				return;

			const float4 surfel_vertex_confid = live_geometry.vertex_confid[idx];
			const float4 surfel_normal_radius = live_geometry.normal_radius[idx];
			const float4 surfel_color_time = live_geometry.color_time[idx];
			// 1. Transfer to camera space
			const float3 vertex = world2camera.rot * surfel_vertex_confid + world2camera.trans;

			// 2. Project to camera image
			const int x = __float2int_rn(((vertex.x / (vertex.z + 1e-10)) * intrinsic.focal_x) + intrinsic.principal_x);
			const int y = __float2int_rn(((vertex.y / (vertex.z + 1e-10)) * intrinsic.focal_y) + intrinsic.principal_y);

			// 3. The corrdinate on fusion_map
			const int map_x_center = scale_factor * x; // +windows_halfsize;
			const int map_y_center = scale_factor * y; // +windows_halfsize;
			int front_counter = 0;

			// 4. Window search
			for (auto map_y = map_y_center - window_halfsize; map_y < map_y_center + window_halfsize; map_y++)
			{
				for (auto map_x = map_x_center - window_halfsize; map_x < map_x_center + window_halfsize; map_x++)
				{
					const float4 map_vertex_confid = tex2D<float4>(fusion_maps.vertex_confid_map, map_x, map_y);
					const float4 map_normal_radius = tex2D<float4>(fusion_maps.normal_radius_map, map_x, map_y);
					const auto index = tex2D<unsigned>(fusion_maps.index_map, map_x, map_y);
					if (index != 0xFFFFFFFF)
					{
						const auto dot_value = dotxyz(surfel_normal_radius, map_normal_radius);
						const float3 diff_camera = world2camera.rot * (surfel_vertex_confid - map_vertex_confid);
						if (diff_camera.z >= 0 && diff_camera.z <= 3 * 0.001 && dot_value >= 0.8)
						{
							front_counter++;
						}
					}
				}
			}

			// 5. Judge removal
			unsigned keep_indicator = 1;

			// 5.1. Check the number of front surfels
			if (front_counter > front_threshold)
				keep_indicator = 0;

			// 5.2. Check the initialize time
			if (surfel_vertex_confid.w < 10.0f && (current_time - initialization_time(surfel_color_time)) > 30.0f)
				keep_indicator = 0;

			// 5.3. (Optional)
			// Single-pixel rejection version
			const float depth4removal = tex2D<float>(measure_maps.depth4removal_map, x, y);
			if (depth4removal - vertex.z > d_differ_outlier_thresh)
				keep_indicator = 0;

			// 5.4. Write to output
			if (keep_indicator == 1 && remaining_surfel[idx] == 0)
			{
				remaining_surfel[idx] = 1;
			}
		}
	};

	__global__ void markRemainingSurfelSurfelWarpKernel(
		const RemainingSurfelMarkerDeviceSurfelWarp marker)
	{
		marker.processMarking();
	}

	__global__ void InitializationSurfelWarpKernel(
		unsigned *indicator,
		const unsigned array_size)
	{
		const auto idx = threadIdx.x + blockDim.x * blockIdx.x;
		if (idx >= array_size)
			return;
		indicator[idx] = 0;
	}

}

star::FusionRemainingSurfelMarker::FusionRemainingSurfelMarker(const unsigned num_cam) : m_num_cam(num_cam), m_num_valid_surfel(0)
{
	for (auto cam_idx = 0; cam_idx < m_num_cam; ++cam_idx)
	{
		m_world2cam[cam_idx] = mat34::identity();
	}
	m_current_time = 0.f;

	m_remaining_surfel_indicator.AllocateBuffer(d_max_num_surfels);
	// The buffer for prefixsum
	m_remaining_indicator_prefixsum.AllocateBuffer(d_max_num_surfels);
	cudaSafeCall(cudaMallocHost((void **)&m_num_remainig_surfel, sizeof(unsigned)));

	// Debug method
	m_remaining_alignment_error.AllocateBuffer(d_max_num_surfels);
}

star::FusionRemainingSurfelMarker::~FusionRemainingSurfelMarker()
{
	m_remaining_surfel_indicator.ReleaseBuffer();
	cudaSafeCall(cudaFreeHost(m_num_remainig_surfel));

	// Debug method
	m_remaining_alignment_error.ReleaseBuffer();
}

void star::FusionRemainingSurfelMarker::SetInputs(
	const Measure4Fusion &measure4fusion,
	const FusionMaps &fusion_maps,
	float current_time,
	const Intrinsic *intrinsic,
	const Extrinsic *cam2world)
{
	m_measure4fusion = measure4fusion;
	m_fusion_maps = fusion_maps;
	m_current_time = current_time;

	for (auto cam_idx = 0; cam_idx < m_num_cam; ++cam_idx)
	{
		m_world2cam[cam_idx] = mat34(cam2world[cam_idx].inverse());
		m_intrinsic[cam_idx] = intrinsic[cam_idx];
	}
}

void star::FusionRemainingSurfelMarker::SetInputs(
	const Measure4GeometryRemoval &measure4geometry_removal,
	const Geometry4Fusion &geometry4fusion,
	const FusionMaps &fusion_maps,
	float current_time,
	const Intrinsic *intrinsic,
	const Extrinsic *cam2world)
{
	m_measure4geometry_removal = measure4geometry_removal;
	m_geometry4fusion = geometry4fusion;
	m_fusion_maps = fusion_maps;
	m_current_time = current_time;

	for (auto cam_idx = 0; cam_idx < m_num_cam; ++cam_idx)
	{
		m_world2cam[cam_idx] = mat34(cam2world[cam_idx].inverse());
		m_intrinsic[cam_idx] = intrinsic[cam_idx];
	}
}

void star::FusionRemainingSurfelMarker::Initialization(const unsigned surfel_size, cudaStream_t stream)
{
	dim3 blk(256);
	dim3 grid(divUp(surfel_size, blk.x));
	device::InitializationKernel<<<grid, blk, 0, stream>>>(
		m_remaining_surfel_indicator,
		m_remaining_alignment_error,
		surfel_size);

	// Sync and check error
#if defined(CUDA_DEBUG_SYNC_CHECK)
	cudaSafeCall(cudaStreamSynchronize(stream));
	cudaSafeCall(cudaGetLastError());
#endif

	m_remaining_surfel_indicator.ResizeArrayOrException(surfel_size);
	m_remaining_alignment_error.ResizeArrayOrException(surfel_size); // Debug
	m_num_valid_surfel = surfel_size;
}

void star::FusionRemainingSurfelMarker::UpdateRemainingSurfelIndicator(cudaStream_t stream)
{
	for (auto cam_idx = 0; cam_idx < m_num_cam; ++cam_idx)
	{
		// Construct the marker
		device::RemainingSurfelMarkerDevice marker;

		marker.observation_maps.vertex_confid_map = m_measure4fusion.vertex_confid_map[cam_idx];
		marker.observation_maps.normal_radius_map = m_measure4fusion.normal_radius_map[cam_idx];
		marker.observation_maps.index_map = m_measure4fusion.index_map[cam_idx];
		marker.observation_maps.color_time_map = m_measure4fusion.color_time_map[cam_idx];

		marker.fusion_maps.vertex_confid_map = m_fusion_maps.vertex_confid_map[cam_idx];
		marker.fusion_maps.normal_radius_map = m_fusion_maps.normal_radius_map[cam_idx];
		marker.fusion_maps.color_time_map = m_fusion_maps.color_time_map[cam_idx];
		marker.fusion_maps.index_map = m_fusion_maps.index_map[cam_idx];

		marker.remaining_surfel = m_remaining_surfel_indicator.Ptr();
		marker.world2camera = m_world2cam[cam_idx];
		marker.intrinsic = m_intrinsic[cam_idx];
		marker.current_time = m_current_time;

		// Debug
		marker.remaining_alignment_error = m_remaining_alignment_error.Ptr();

		unsigned fusion_img_rows, fusion_img_cols;
		query2DTextureExtent(m_fusion_maps.vertex_confid_map[cam_idx], fusion_img_cols, fusion_img_rows);
		marker.img_cols = fusion_img_cols;
		marker.img_rows = fusion_img_rows;

		dim3 blk(16, 16);
		dim3 grid(divUp(fusion_img_cols, blk.x), divUp(fusion_img_rows, blk.y));
		device::updateRemainingSurfelKernel<<<grid, blk, 0, stream>>>(marker);
	}

	// Sync and check error
#if defined(CUDA_DEBUG_SYNC_CHECK)
	cudaSafeCall(cudaStreamSynchronize(stream));
	cudaSafeCall(cudaGetLastError());
#endif
}

void star::FusionRemainingSurfelMarker::PostProcessRemainingSurfelIndicator(cudaStream_t stream)
{
	dim3 blk(256);
	dim3 grid(divUp(m_num_valid_surfel, blk.x));

	device::PostProcessMarking<<<grid, blk, 0, stream>>>(
		m_remaining_alignment_error.Ptr(),
		m_remaining_surfel_indicator.Ptr(),
		m_num_valid_surfel);

	// Sync and check error
#if defined(CUDA_DEBUG_SYNC_CHECK)
	cudaSafeCall(cudaStreamSynchronize(stream));
	cudaSafeCall(cudaGetLastError());
#endif
}

void star::FusionRemainingSurfelMarker::RemainingSurfelIndicatorPrefixSumSync(cudaStream_t stream)
{
	m_remaining_indicator_prefixsum.InclusiveSum(m_remaining_surfel_indicator.View(), stream);
	cudaSafeCall(cudaMemcpyAsync(
		(void *)m_num_remainig_surfel,
		m_remaining_indicator_prefixsum.valid_prefixsum_array + m_remaining_indicator_prefixsum.valid_prefixsum_array.size() - 1,
		sizeof(unsigned),
		cudaMemcpyDeviceToHost,
		stream));
	// cudaSafeCall(cudaStreamSynchronize(stream)); // This stream is not working....
	cudaSafeCall(cudaDeviceSynchronize()); // FIXME: I have not CLUE on why I have to use this!!!!
										   // std::cout << "Sum: Remaining: " << (*m_num_remainig_surfel) << std::endl;
}

star::GArrayView<unsigned int>
star::FusionRemainingSurfelMarker::GetRemainingSurfelIndicatorPrefixsum() const
{
	const auto &prefixsum_array = m_remaining_indicator_prefixsum.valid_prefixsum_array;
	STAR_CHECK(m_remaining_surfel_indicator.ArraySize() == prefixsum_array.size());
	return GArrayView<unsigned>(prefixsum_array);
}

/* SurfelWarp version operation
 */
void star::FusionRemainingSurfelMarker::InitializationSurfelWarp(const unsigned surfel_size, cudaStream_t stream)
{
	dim3 blk(256);
	dim3 grid(divUp(surfel_size, blk.x));
	device::InitializationSurfelWarpKernel<<<grid, blk, 0, stream>>>(
		m_remaining_surfel_indicator,
		surfel_size);

	// Sync and check error
#if defined(CUDA_DEBUG_SYNC_CHECK)
	cudaSafeCall(cudaStreamSynchronize(stream));
	cudaSafeCall(cudaGetLastError());
#endif
	m_remaining_surfel_indicator.ResizeArrayOrException(surfel_size);
	m_remaining_alignment_error.ResizeArrayOrException(surfel_size); // Debug
	m_num_valid_surfel = surfel_size;
}

void star::FusionRemainingSurfelMarker::UpdateRemainingSurfelIndicatorSurfelWarp(cudaStream_t stream)
{
	for (auto cam_idx = 0; cam_idx < m_num_cam; ++cam_idx)
	{
		// Construct the marker
		device::RemainingSurfelMarkerDeviceSurfelWarp marker;

		marker.fusion_maps.vertex_confid_map = m_fusion_maps.vertex_confid_map[cam_idx];
		marker.fusion_maps.normal_radius_map = m_fusion_maps.normal_radius_map[cam_idx];
		marker.fusion_maps.color_time_map = m_fusion_maps.color_time_map[cam_idx];
		marker.fusion_maps.index_map = m_fusion_maps.index_map[cam_idx];

		marker.live_geometry.vertex_confid = m_geometry4fusion.vertex_confid.View();
		marker.live_geometry.normal_radius = m_geometry4fusion.normal_radius.Ptr();
		marker.live_geometry.color_time = m_geometry4fusion.color_time.Ptr();

		marker.measure_maps.depth4removal_map = m_measure4geometry_removal.depth4removal_map[cam_idx];

		marker.remaining_surfel = m_remaining_surfel_indicator.Ptr();
		marker.world2camera = m_world2cam[cam_idx];
		marker.intrinsic = m_intrinsic[cam_idx];
		marker.current_time = m_current_time;

		// Sanity check
		STAR_CHECK_NE(m_geometry4fusion.vertex_confid.Size(), 0);
		std::cout << "Geometry size: " << m_geometry4fusion.vertex_confid.Size() << std::endl;

		dim3 blk(256);
		dim3 grid(divUp(m_geometry4fusion.vertex_confid.Size(), blk.x));
		device::markRemainingSurfelSurfelWarpKernel<<<grid, blk, 0, stream>>>(marker);
	}

	// Sync and check error
#if defined(CUDA_DEBUG_SYNC_CHECK)
	cudaSafeCall(cudaStreamSynchronize(stream));
	cudaSafeCall(cudaGetLastError());
#endif
}