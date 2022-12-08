#include <star/common/types/vecX_op.h>
#include <star/geometry/surfel/surfel_fusion_dev.cuh>
#include "SurfelFusionHandler.h"

namespace star::device
{

	struct FusionAndMarkAppendedObservationSurfelDevice
	{
		// Some constants defined as enum
		enum
		{
			scale_factor = d_fusion_map_scale,
			fuse_window_halfsize = scale_factor >> 1,
			count_model_halfsize = 2 * scale_factor /*>> 1 */,
			append_window_halfsize = scale_factor,
			search_window_halfsize = scale_factor,
		};

		// The observation
		struct
		{
			cudaTextureObject_t vertex_time_map;
			cudaTextureObject_t normal_radius_map;
			cudaTextureObject_t color_time_map;
			cudaTextureObject_t index_map;
		} measure_maps;

		// The semantic map
		struct
		{
			cudaTextureObject_t semantic_map;
		} semantic_maps;

		// The rendered maps
		struct
		{
			cudaTextureObject_t vertex_map;
			cudaTextureObject_t normal_map;
			cudaTextureObject_t color_time_map;
			cudaTextureObject_t index_map;
		} render_maps;

		// The written array
		struct
		{
			float4 *vertex_confid;
			float4 *normal_radius;
			float4 *color_time;
			unsigned *fused_indicator;
			ucharX<d_max_num_semantic> *semantic_prob;
		} geometry_arrays;

		// The shared datas
		unsigned short image_rows, image_cols;
		float current_time;

		__host__ __device__ __forceinline__ bool directionConsistency(
			const float4 &depth_vertex, const float4 &depth_normal, const float threshold = 0.4f) const
		{
			const float3 view_direction = -normalized(make_float3(depth_vertex.x, depth_vertex.y, depth_vertex.z));
			const float3 normal = normalized(make_float3(depth_normal.x, depth_normal.y, depth_normal.z));
			return dot(view_direction, normal) > threshold;
		}

		// The actual processing interface
		__device__ __forceinline__ void processIndicator(const mat34 &world2camera, unsigned *appending_indicator, const bool update_semantic) const
		{
			const int x = threadIdx.x + blockDim.x * blockIdx.x;
			const int y = threadIdx.y + blockDim.y * blockIdx.y;
			if (x < search_window_halfsize || x >= image_cols - search_window_halfsize || y < search_window_halfsize || y >= image_rows - search_window_halfsize)
			{
				return;
			}

			// 1. Load the measure surfel
			const float4 measure_vertex_confid = tex2D<float4>(measure_maps.vertex_time_map, x, y);
			const float4 measure_normal_radius = tex2D<float4>(measure_maps.normal_radius_map, x, y);
			const float4 measure_color_time = tex2D<float4>(measure_maps.color_time_map, x, y);
			const unsigned measure_index = tex2D<unsigned>(measure_maps.index_map, x, y);
			if (is_zero_vertex(measure_vertex_confid))
				return;

			// 2. Prepare for windows search
			const int map_x_center = scale_factor * x;
			const int map_y_center = scale_factor * y;
			SurfelFusionWindowState fusion_state;
			bool matched_in_window = false;
			SurfelAppendingWindowState append_state;
			unsigned model_count = 0;

			// 3. Start window search
			for (int dy = -search_window_halfsize; dy < search_window_halfsize; dy++)
			{
				for (int dx = -search_window_halfsize; dx < search_window_halfsize; dx++)
				{
					const int map_y = dy + map_y_center;
					const int map_x = dx + map_x_center;
					const auto geometry_index = tex2D<unsigned>(render_maps.index_map, map_x, map_y);

					if (geometry_index != 0xFFFFFFFF)
					{
						// 3.1. Load the model vertex
						const float4 model_world_v4 = tex2D<float4>(render_maps.vertex_map, map_x, map_y);
						const float4 model_world_n4 = tex2D<float4>(render_maps.normal_map, map_x, map_y);

						// 3.2. Transform model vertex to camera coordinate
						const float3 model_camera_v3 = world2camera.rot * model_world_v4 + world2camera.trans;
						const float3 model_camera_n3 = world2camera.rot * model_world_n4;

						// 3.3. Compute some attributes commonly used for checking
						const float dot_value = dotxyz(model_camera_n3, measure_normal_radius);
						const float diff_z = fabsf(model_camera_v3.z - measure_vertex_confid.z);
						const float confidence = model_world_v4.w;
						const float z_dist = model_camera_v3.z;
						const float dist_square = squared_distance(model_camera_v3, measure_vertex_confid);

						// 3.4. Check fusion status
						if (dx >= -fuse_window_halfsize && dy >= -fuse_window_halfsize && dx < fuse_window_halfsize && dy < fuse_window_halfsize)
						{
							if (dot_value >= 0.8f && diff_z <= 3 * 0.001f)
							{ // Update it
								fusion_state.Update(confidence, z_dist, map_x, map_y);
							}
						}

						// 3.5. Check for matched in window
						if (dx >= -count_model_halfsize && dy >= -count_model_halfsize && dx < count_model_halfsize && dy < count_model_halfsize)
						{
							if (dot_value > 0.3f)
								model_count++;
						}

						// 3.6. Check for appending
						{
							if (dot_value >= 0.8f && dist_square <= (2 * 0.001f) * (2 * 0.001f))
							{ // Update it
								append_state.Update(confidence, z_dist);
							}
						}
					} // There is a surfel here
				}	  // x iteration loop
			}		  // y iteration loop

			// 4. Mark appending status
			if (append_state.best_confid < -0.01 && model_count == 0 && directionConsistency(measure_vertex_confid, measure_normal_radius, 0.4f))
			{
				atomicOr(&appending_indicator[measure_index], (unsigned)1);
			}

			// 5. Fusion
			if (fusion_state.best_confid > 0)
			{
				// 5.1. Get best matched geometry
				float4 model_vertex_confid = tex2D<float4>(render_maps.vertex_map, fusion_state.best_map_x, fusion_state.best_map_y);
				float4 model_normal_radius = tex2D<float4>(render_maps.normal_map, fusion_state.best_map_x, fusion_state.best_map_y);
				float4 model_color_time = tex2D<float4>(render_maps.color_time_map, fusion_state.best_map_x, fusion_state.best_map_y);
				const unsigned geometry_index = tex2D<unsigned>(render_maps.index_map, fusion_state.best_map_x, fusion_state.best_map_y);

				// 5.2. Fusion
				fuse_surfel(
					measure_vertex_confid, measure_normal_radius, measure_color_time,
					world2camera, current_time,
					model_vertex_confid, model_normal_radius, model_color_time);

				// fuse_surfel_stay_color(
				//	measure_vertex_confid, measure_normal_radius, measure_color_time,
				//	world2camera, current_time,
				//	model_vertex_confid, model_normal_radius, model_color_time
				//);

				// fuse_surfel_replace_color(
				//	measure_vertex_confid, measure_normal_radius, measure_color_time,
				//	world2camera, current_time,
				//	model_vertex_confid, model_normal_radius, model_color_time
				//);

				// fuse_surfel_stay_or_replace_color(
				//	measure_vertex_confid, measure_normal_radius, measure_color_time,
				//	world2camera, current_time,
				//	model_vertex_confid, model_normal_radius, model_color_time,
				//	100
				//);

				// fuse_surfel_clip_color(
				//	measure_vertex_confid, measure_normal_radius, measure_color_time,
				//	world2camera, current_time,
				//	model_vertex_confid, model_normal_radius, model_color_time,
				//	40
				//);

				// 5.2.5 (Optional) Semantic Fusion
				if (update_semantic)
				{
					int measure_semantic = tex2D<int>(semantic_maps.semantic_map, x, y);
					ucharX<d_max_num_semantic> model_semantic = geometry_arrays.semantic_prob[geometry_index];
					fuse_surfel_semantic(
						measure_semantic,
						model_semantic);
					geometry_arrays.semantic_prob[geometry_index] = model_semantic;
				}

				// FIXME: Is it possible that there is a conflict here? Upsampling leverage such conflict
				// 5.3. Write to buffer
				geometry_arrays.vertex_confid[geometry_index] = model_vertex_confid;
				geometry_arrays.normal_radius[geometry_index] = model_normal_radius;
				geometry_arrays.color_time[geometry_index] = model_color_time;
				geometry_arrays.fused_indicator[geometry_index] = 1;
			}
		}

		// The fusion processor for re-initialize
		__device__ __forceinline__ void processFusionReinit(const mat34 &world2camera, unsigned *appending_indicator) const
		{
			const int x = threadIdx.x + blockDim.x * blockIdx.x;
			const int y = threadIdx.y + blockDim.y * blockIdx.y;
			const auto offset = y * image_cols + x;
			if (x < search_window_halfsize || x >= image_cols - search_window_halfsize || y < search_window_halfsize || y >= image_rows - search_window_halfsize)
			{
				return;
			}

			// 1. Load the measure surfel
			const float4 measure_vertex_confid = tex2D<float4>(measure_maps.vertex_time_map, x, y);
			const float4 measure_normal_radius = tex2D<float4>(measure_maps.normal_radius_map, x, y);
			const float4 measure_color_time = tex2D<float4>(measure_maps.color_time_map, x, y);
			const unsigned measure_index = tex2D<unsigned>(measure_maps.index_map, x, y);
			if (is_zero_vertex(measure_vertex_confid))
				return;

			// 2. Prepare for windows search
			const int map_x_center = scale_factor * x;
			const int map_y_center = scale_factor * y;
			SurfelFusionWindowState fusion_state;

			// 3. Start window search
			for (int dy = -fuse_window_halfsize; dy < fuse_window_halfsize; dy++)
			{
				for (int dx = -fuse_window_halfsize; dx < fuse_window_halfsize; dx++)
				{
					// The actual position of in the rendered map
					const int map_y = dy + map_y_center;
					const int map_x = dx + map_x_center;

					const auto geometry_index = tex2D<unsigned>(render_maps.index_map, map_x, map_y);
					if (geometry_index != 0xFFFFFFFF)
					{
						// 3.1. Load the model vertex
						const float4 model_world_v4 = tex2D<float4>(render_maps.vertex_map, map_x, map_y);
						const float4 model_world_n4 = tex2D<float4>(render_maps.normal_map, map_x, map_y);

						// 3.2. Transform model vertex to camera coordinate
						const float3 model_camera_v3 = world2camera.rot * model_world_v4 + world2camera.trans;
						const float3 model_camera_n3 = world2camera.rot * model_world_n4;

						// 3.3. Compute some attributes commonly used for checking
						const float dot_value = dotxyz(model_camera_n3, measure_normal_radius);
						const float diff_z = fabsf(model_camera_v3.z - measure_vertex_confid.z);
						const float confidence = model_world_v4.w;
						const float z_dist = model_camera_v3.z;

						// 3.4. Check fusion status
						if (dot_value >= 0.9f && diff_z <= 2 * 0.001f)
						{ // Update it
							fusion_state.Update(confidence, z_dist, map_x, map_y);
						}
					} // There is a surfel here
				}	  // x iteration loop
			}		  // y iteration loop

			// 4. For appending, as in reinit should mark all depth surfels
			if (fusion_state.best_confid < -0.01)
			{
				atomicOr(&appending_indicator[measure_index], (unsigned)1);
			}

			// 5. Fusion
			if (fusion_state.best_confid > 0)
			{
				// 5.1. Get best matched geometry
				float4 model_vertex_confid = tex2D<float4>(render_maps.vertex_map, fusion_state.best_map_x, fusion_state.best_map_y);
				float4 model_normal_radius = tex2D<float4>(render_maps.normal_map, fusion_state.best_map_x, fusion_state.best_map_y);
				float4 model_color_time = tex2D<float4>(render_maps.color_time_map, fusion_state.best_map_x, fusion_state.best_map_y);
				const unsigned geometry_index = tex2D<unsigned>(render_maps.index_map, fusion_state.best_map_x, fusion_state.best_map_y);

				// 5.2. Fusion with replace color
				fuse_surfel_replace_color(
					measure_vertex_confid, measure_normal_radius, measure_color_time,
					world2camera, current_time,
					model_vertex_confid, model_normal_radius, model_color_time);

				// 5.2.5. (Optional) Semantic Fusion
				// TODO

				// 5.3. Write it
				geometry_arrays.vertex_confid[geometry_index] = model_vertex_confid;
				geometry_arrays.normal_radius[geometry_index] = model_normal_radius;
				geometry_arrays.color_time[geometry_index] = model_color_time;
				geometry_arrays.fused_indicator[geometry_index] = 1;
			}
		}
	};

	__global__ void fuseAndMarkAppendedObservationSurfelsKernel(
		const FusionAndMarkAppendedObservationSurfelDevice fuser,
		mat34 world2camera,
		unsigned *appended_pixel,
		const bool update_semantic)
	{
		fuser.processIndicator(world2camera, appended_pixel, update_semantic);
	}

	__global__ void fuseAndMarkAppendedObservationSurfelReinitKernel(
		const FusionAndMarkAppendedObservationSurfelDevice fuser,
		mat34 world2camera,
		unsigned *appended_pixel)
	{
		fuser.processFusionReinit(world2camera, appended_pixel);
	}

	__global__ void CompactAppendCandidateVertexKernel(
		const unsigned *append_indicator,
		const unsigned *append_indicator_prefixsum,
		const float4 *vertex_confid_array,
		const float4 *normal_radius_array,
		const float4 *color_time_array,
		float4 *append_candidate_vertex,
		float4 *append_candidate_normal,
		float4 *append_color_time,
		const unsigned surfel_size)
	{
		const int idx = threadIdx.x + blockDim.x * blockIdx.x;
		if (idx >= surfel_size)
			return;

		if (append_indicator[idx])
		{
			const auto offset = append_indicator_prefixsum[idx] - 1;
			append_candidate_vertex[offset] = vertex_confid_array[idx];
			append_candidate_normal[offset] = normal_radius_array[idx];
			append_color_time[offset] = color_time_array[idx];
		}
	}

	// With semantic
	__global__ void CompactAppendCandidateVertexKernel(
		const unsigned *append_indicator,
		const unsigned *append_indicator_prefixsum,
		const float4 *vertex_confid_array,
		const float4 *normal_radius_array,
		const float4 *color_time_array,
		const ucharX<d_max_num_semantic> *semantic_prob_array,
		float4 *append_candidate_vertex,
		float4 *append_candidate_normal,
		float4 *append_color_time,
		ucharX<d_max_num_semantic> *append_semantic_prob,
		const unsigned surfel_size)
	{
		const int idx = threadIdx.x + blockDim.x * blockIdx.x;
		if (idx >= surfel_size)
			return;

		if (append_indicator[idx])
		{
			const auto offset = append_indicator_prefixsum[idx] - 1;
			append_candidate_vertex[offset] = vertex_confid_array[idx];
			append_candidate_normal[offset] = normal_radius_array[idx];
			append_color_time[offset] = color_time_array[idx];
			append_semantic_prob[offset] = semantic_prob_array[idx];
		}
	}

}

star::SurfelFusionHandler::SurfelFusionHandler(
	const unsigned num_cam,
	const unsigned *img_cols,
	const unsigned *img_rows,
	const bool enable_semantic_surfel) : m_num_cam(num_cam), m_current_time(0.f), m_num_appended_surfel(0),
										 m_enable_semantic_surfel(enable_semantic_surfel)
{
	// The surfel indicator is in the size of maximun surfels
	m_remaining_surfel_indicator.AllocateBuffer(d_max_num_surfels);

	m_appended_surfel_vertex.AllocateBuffer(d_max_num_surfels);
	m_appended_surfel_normal.AllocateBuffer(d_max_num_surfels);
	m_appended_surfel_color_time.AllocateBuffer(d_max_num_surfels);
	m_appended_surfel_semantic_prob.AllocateBuffer(d_max_num_surfels);
	// The append depth indicator is always in the same size as image pixels
	m_appended_surfel_indicator_prefixsum.AllocateBuffer(d_max_num_surfels);
	m_appended_surfel_indicator.AllocateBuffer(d_max_num_surfels);

	// The rows and cols of the image
	for (auto cam_idx = 0; cam_idx < m_num_cam; ++cam_idx)
	{
		m_image_cols[cam_idx] = img_cols[cam_idx];
		m_image_rows[cam_idx] = img_rows[cam_idx];
	}
}

star::SurfelFusionHandler::~SurfelFusionHandler()
{
	m_appended_surfel_vertex.ReleaseBuffer();
	m_appended_surfel_normal.ReleaseBuffer();
	m_appended_surfel_color_time.ReleaseBuffer();
	m_appended_surfel_semantic_prob.ReleaseBuffer();
	m_remaining_surfel_indicator.ReleaseBuffer();
	m_appended_surfel_indicator.ReleaseBuffer();
}

void star::SurfelFusionHandler::SetInputs(
	const FusionMaps &fusion_maps,
	const Measure4Fusion &measure4fusion,
	Geometry4Fusion &geometry4fusion,
	float current_time,
	const Extrinsic *cam2world)
{
	m_fusion_maps = fusion_maps;
	m_geometry4fusion = geometry4fusion;
	m_measure4fusion = measure4fusion;
	m_current_time = current_time;
	for (auto cam_idx = 0; cam_idx < m_num_cam; ++cam_idx)
	{
		m_world2cam[cam_idx] = mat34(cam2world[cam_idx].inverse());
	}
}

void star::SurfelFusionHandler::SetInputs(
	const FusionMaps &fusion_maps,
	const Measure4Fusion &measure4fusion,
	const Segmentation4SemanticFusion &segmentation4semantic_fusion,
	Geometry4Fusion &geometry4fusion,
	Geometry4SemanticFusion &geometry4semantic_fusion,
	float current_time,
	const Extrinsic *cam2world)
{
	m_fusion_maps = fusion_maps;
	m_geometry4fusion = geometry4fusion;
	m_measure4fusion = measure4fusion;
	// Optional
	m_segmentation4semantic_fusion = segmentation4semantic_fusion;
	m_geometry4semantic_fusion = geometry4semantic_fusion;
	m_current_time = current_time;
	for (auto cam_idx = 0; cam_idx < m_num_cam; ++cam_idx)
	{
		m_world2cam[cam_idx] = mat34(cam2world[cam_idx].inverse());
	}
}

void star::SurfelFusionHandler::ZeroInitializeIndicator(
	const unsigned num_geometry_surfels, const unsigned num_observation_surfels, cudaStream_t stream)
{
	// Appending
	cudaSafeCall(cudaMemsetAsync(
		m_appended_surfel_indicator.Ptr(),
		0, num_observation_surfels * sizeof(unsigned),
		stream));
	m_appended_surfel_indicator.ResizeArrayOrException(num_observation_surfels);

	// Remaining
	cudaSafeCall(cudaMemsetAsync(
		m_remaining_surfel_indicator.Ptr(),
		0, num_geometry_surfels * sizeof(unsigned),
		stream));
	m_remaining_surfel_indicator.ResizeArrayOrException(num_geometry_surfels);
}

star::GArraySlice<unsigned> star::SurfelFusionHandler::GetRemainingSurfelIndicator()
{
	return m_remaining_surfel_indicator.Slice();
}

void star::SurfelFusionHandler::ProcessFusion(const bool update_semantic, cudaStream_t stream)
{
	// Do fusion & labeling
	processFusionAndAppendLabel(update_semantic, stream);

	// Sync and check error
#if defined(CUDA_DEBUG_SYNC_CHECK)
	cudaSafeCall(cudaStreamSynchronize(stream));
	cudaSafeCall(cudaGetLastError());
#endif
}

void star::SurfelFusionHandler::ProcessFusionReinit(cudaStream_t stream)
{
	processFusionReinit(stream);

	// Sync and check error
#if defined(CUDA_DEBUG_SYNC_CHECK)
	cudaSafeCall(cudaStreamSynchronize(stream));
	cudaSafeCall(cudaGetLastError());
#endif
}

void star::SurfelFusionHandler::prepareFuserArguments(const bool update_semantic, const unsigned cam_idx, void *fuser_ptr)
{
	// Recovery the fuser arguments
	auto &fuser = *((device::FusionAndMarkAppendedObservationSurfelDevice *)fuser_ptr);

	// The observation maps
	fuser.measure_maps.vertex_time_map = m_measure4fusion.vertex_confid_map[cam_idx];
	fuser.measure_maps.normal_radius_map = m_measure4fusion.normal_radius_map[cam_idx];
	fuser.measure_maps.color_time_map = m_measure4fusion.color_time_map[cam_idx];
	fuser.measure_maps.index_map = m_measure4fusion.index_map[cam_idx];

	// The semantic maps
	if (update_semantic)
	{
		fuser.semantic_maps.semantic_map = m_segmentation4semantic_fusion.segmentation[cam_idx];
		fuser.geometry_arrays.semantic_prob = m_geometry4semantic_fusion.semantic_prob.Ptr();
	}

	// The rendered maps
	fuser.render_maps.vertex_map = m_fusion_maps.vertex_confid_map[cam_idx];
	fuser.render_maps.normal_map = m_fusion_maps.normal_radius_map[cam_idx];
	fuser.render_maps.color_time_map = m_fusion_maps.color_time_map[cam_idx];
	fuser.render_maps.index_map = m_fusion_maps.index_map[cam_idx];

	// The written array
	fuser.geometry_arrays.vertex_confid = m_geometry4fusion.vertex_confid.Ptr();
	fuser.geometry_arrays.normal_radius = m_geometry4fusion.normal_radius.Ptr();
	fuser.geometry_arrays.color_time = m_geometry4fusion.color_time.Ptr();
	fuser.geometry_arrays.fused_indicator = m_remaining_surfel_indicator.Ptr();

	// Other attributes
	fuser.current_time = m_current_time;
	fuser.image_cols = m_image_cols[cam_idx];
	fuser.image_rows = m_image_rows[cam_idx];
}

void star::SurfelFusionHandler::processFusionAndAppendLabel(const bool update_semantic, cudaStream_t stream)
{
	for (auto cam_idx = 0; cam_idx < d_max_cam; ++cam_idx)
	{
		// Construct the fuser
		device::FusionAndMarkAppendedObservationSurfelDevice fuser;
		prepareFuserArguments(update_semantic, cam_idx, (void *)&fuser);

		dim3 blk(16, 16);
		dim3 grid(divUp(m_image_cols[cam_idx], blk.x), divUp(m_image_rows[cam_idx], blk.y));
		device::fuseAndMarkAppendedObservationSurfelsKernel<<<grid, blk, 0, stream>>>(
			fuser,
			m_world2cam[cam_idx],
			m_appended_surfel_indicator,
			update_semantic);
	}

#if defined(CUDA_DEBUG_SYNC_CHECK)
	cudaSafeCall(cudaStreamSynchronize(stream));
	cudaSafeCall(cudaGetLastError());
#endif
}

void star::SurfelFusionHandler::processFusionReinit(const bool update_semantic, cudaStream_t stream)
{
	for (auto cam_idx = 0; cam_idx < d_max_cam; ++cam_idx)
	{
		// Resize the array
		const auto num_surfels = m_geometry4fusion.vertex_confid.Size();
		m_remaining_surfel_indicator.ResizeArrayOrException(num_surfels);

		// Construct the fuser
		device::FusionAndMarkAppendedObservationSurfelDevice fuser;
		prepareFuserArguments(update_semantic, cam_idx, (void *)&fuser);

		dim3 blk(16, 16);
		dim3 grid(divUp(m_image_cols[cam_idx], blk.x), divUp(m_image_rows[cam_idx], blk.y));
		device::fuseAndMarkAppendedObservationSurfelReinitKernel<<<grid, blk, 0, stream>>>(
			fuser,
			m_world2cam[cam_idx],
			m_appended_surfel_indicator);
	}

#if defined(CUDA_DEBUG_SYNC_CHECK)
	cudaSafeCall(cudaStreamSynchronize(stream));
	cudaSafeCall(cudaGetLastError());
#endif
}

void star::SurfelFusionHandler::CompactAppendedCandidate(
	const GArrayView<float4> &vertex_config_array,
	const GArrayView<float4> &normal_radius_array,
	const GArrayView<float4> &color_time_array,
	cudaStream_t stream)
{
	m_appended_surfel_indicator_prefixsum.InclusiveSum(m_appended_surfel_indicator.View(), stream);
	// Query the size
	cudaSafeCall(cudaMemcpyAsync(
		&m_num_appended_surfel,
		m_appended_surfel_indicator_prefixsum.valid_prefixsum_array.ptr() + m_appended_surfel_indicator_prefixsum.valid_prefixsum_array.size() - 1,
		sizeof(unsigned),
		cudaMemcpyDeviceToHost,
		stream));

	cudaSafeCall(cudaStreamSynchronize(stream));
	m_appended_surfel_vertex.ResizeArrayOrException(m_num_appended_surfel);
	m_appended_surfel_normal.ResizeArrayOrException(m_num_appended_surfel);
	m_appended_surfel_color_time.ResizeArrayOrException(m_num_appended_surfel);

	if (m_num_appended_surfel == 0)
		return; // No appending, early stop

	const auto num_surfels = vertex_config_array.Size();

	dim3 blk(128);
	dim3 grid(divUp(num_surfels, blk.x));

	device::CompactAppendCandidateVertexKernel<<<grid, blk, 0, stream>>>(
		m_appended_surfel_indicator,
		m_appended_surfel_indicator_prefixsum.valid_prefixsum_array,
		vertex_config_array,
		normal_radius_array,
		color_time_array,
		m_appended_surfel_vertex,
		m_appended_surfel_normal,
		m_appended_surfel_color_time,
		num_surfels);

	// Sync and check error
#if defined(CUDA_DEBUG_SYNC_CHECK)
	cudaSafeCall(cudaStreamSynchronize(stream));
	cudaSafeCall(cudaGetLastError());
#endif
}

void star::SurfelFusionHandler::CompactAppendedCandidate(
	const GArrayView<float4> &vertex_config_array,
	const GArrayView<float4> &normal_radius_array,
	const GArrayView<float4> &color_time_array,
	const GArrayView<ucharX<d_max_num_semantic>> &semantic_prob_array,
	cudaStream_t stream)
{
	m_appended_surfel_indicator_prefixsum.InclusiveSum(m_appended_surfel_indicator.View(), stream);
	// Query the size
	cudaSafeCall(cudaMemcpyAsync(
		&m_num_appended_surfel,
		m_appended_surfel_indicator_prefixsum.valid_prefixsum_array.ptr() + m_appended_surfel_indicator_prefixsum.valid_prefixsum_array.size() - 1,
		sizeof(unsigned),
		cudaMemcpyDeviceToHost,
		stream));

	cudaSafeCall(cudaStreamSynchronize(stream));
	m_appended_surfel_vertex.ResizeArrayOrException(m_num_appended_surfel);
	m_appended_surfel_normal.ResizeArrayOrException(m_num_appended_surfel);
	m_appended_surfel_color_time.ResizeArrayOrException(m_num_appended_surfel);
	m_appended_surfel_semantic_prob.ResizeArrayOrException(m_num_appended_surfel);

	if (m_num_appended_surfel == 0)
		return; // No appending, early stop

	const auto num_surfels = vertex_config_array.Size();

	dim3 blk(128);
	dim3 grid(divUp(num_surfels, blk.x));

	device::CompactAppendCandidateVertexKernel<<<grid, blk, 0, stream>>>(
		m_appended_surfel_indicator,
		m_appended_surfel_indicator_prefixsum.valid_prefixsum_array,
		vertex_config_array,
		normal_radius_array,
		color_time_array,
		semantic_prob_array,
		m_appended_surfel_vertex,
		m_appended_surfel_normal,
		m_appended_surfel_color_time,
		m_appended_surfel_semantic_prob,
		num_surfels);
	// Sync and check error
#if defined(CUDA_DEBUG_SYNC_CHECK)
	cudaSafeCall(cudaStreamSynchronize(stream));
	cudaSafeCall(cudaGetLastError());
#endif
}