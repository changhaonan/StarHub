#include <mono_star/opt/DenseImageHandler.h>
#include <star/opt/geometry_jacobian.cuh>
#include <device_launch_parameters.h>

namespace star::device
{
	/**
	 * Plane2Plane ICP loss, all vertex, normal are defined in the same coordinate
	 * (e.g. camera or world).
	 * Note: only make sense, when error is small.
	 */
	__device__ void computePointToPlanceICPTermJacobianResidual(
		const float3 &target_vertex,
		const float3 &target_normal,
		const float3 &source_vertex,
		// The output
		GradientOfDenseImage &gradient,
		floatX<d_dense_image_residual_dim> &residual,
		const unsigned offset)
	{
		// Compute the residual terms
		residual[offset] = dot(target_normal, source_vertex - target_vertex);
		gradient.gradient[offset].rotation = cross(source_vertex, target_normal);
		gradient.gradient[offset].translation = target_normal;
	}

	/**
	 * OpticalFlow 2D loss:
	 * The source_vertex is supposed to projected to target_pixel.
	 * The 2D loss is defined on 2D distance.
	 * Note: only make sense, when error is small.
	 */
	__device__ void computeOpticalFlow2DTermJacobianResidual(
		const float2 &target_pixel,
		const float3 &source_vertex,
		const mat34 &world2camera,
		const mat23 &intrinsic,
		// The output
		GradientOfDenseImage &gradient,
		floatX<d_dense_image_residual_dim> &residual,
		const unsigned offset)
	{
		const float3 source_vertex_camera = world2camera.apply_se3(source_vertex);
		const float inv_depth_source_vertex = 1.f / source_vertex_camera.z;
		const float2 source_pixel = intrinsic * (source_vertex_camera * inv_depth_source_vertex);
		const float2 pixel_diff = source_pixel - target_pixel;
		if (norm(pixel_diff) < 0.1f)
		{
			// Ignore small offset
			residual[offset] = 0.f;
			residual[offset + 1] = 0.f;
			gradient.gradient[offset].translation = make_float3(0.f, 0.f, 0.f);
			gradient.gradient[offset + 1].translation = make_float3(0.f, 0.f, 0.f);
			gradient.gradient[offset].rotation = make_float3(0.f, 0.f, 0.f);
			gradient.gradient[offset + 1].rotation = make_float3(0.f, 0.f, 0.f);
		}
		else
		{
			residual[offset] = pixel_diff.x;
			residual[offset + 1] = pixel_diff.y;

			const mat23 intrinsic_from_world = (intrinsic * world2camera.rot) * inv_depth_source_vertex;
			gradient.gradient[offset].translation = intrinsic_from_world.row(0);
			gradient.gradient[offset + 1].translation = intrinsic_from_world.row(1);
			twist_rotation twist_source(-source_vertex);
			const mat23 gradient_rot = (intrinsic_from_world * twist_source.so3());
			gradient.gradient[offset].rotation = gradient_rot.row(0);
			gradient.gradient[offset + 1].rotation = gradient_rot.row(1);
		}
	}

	/**
	 * Modified OpticalFlow 2D loss:
	 * Instead of computing pixel difference, compute the difference of vertex within the camera plane
	 * Note: only make sense, when error is small.
	 */
	__device__ void computeModifiedOpticalFlow2DTermJacobianResidual(
		const float3 &target_vertex_world,
		const float3 &source_vertex_world,
		const mat23 &project_rot_world2camera,
		// The output
		GradientOfDenseImage &gradient,
		floatX<d_dense_image_residual_dim> &residual,
		const unsigned offset

	)
	{
		// 1. Residual
		const float3 diff_vertex = (source_vertex_world - target_vertex_world);
		const float2 e = project_rot_world2camera * diff_vertex;

		if (norm(e) < 1e-6f)
		{
			// 2.1. Ignore small offset
			residual[offset] = 0.f;
			residual[offset + 1] = 0.f;
			gradient.gradient[offset].translation = make_float3(0.f, 0.f, 0.f);
			gradient.gradient[offset + 1].translation = make_float3(0.f, 0.f, 0.f);
			gradient.gradient[offset].rotation = make_float3(0.f, 0.f, 0.f);
			gradient.gradient[offset + 1].rotation = make_float3(0.f, 0.f, 0.f);
		}
		else
		{
			residual[offset] = e.x;
			residual[offset + 1] = e.y;
			// 2.1. J_t
			gradient.gradient[offset].translation = project_rot_world2camera.row(0);
			gradient.gradient[offset + 1].translation = project_rot_world2camera.row(1);
			// 2.2. J_r
			twist_rotation twist_source(-source_vertex_world);
			const mat23 gradient_rot = (project_rot_world2camera * twist_source.so3());
			gradient.gradient[offset].rotation = gradient_rot.row(0);
			gradient.gradient[offset + 1].rotation = gradient_rot.row(1);
		}
	}

	/**
	 * Picp & OpticalFlow loss:
	 * Hybrid loss of all mentioned above.
	 * Weight2D is the relative weight of opticalflow to picp (default as 1)
	 * source, target are all defined in world coordinate
	 */
	__device__ void computeDenseImageTermJacobianResidual(
		const float3 &target_vertex,
		const float3 &target_normal,
		const float2 &target_pixel,
		const float3 &source_vertex,
		const mat34 &world2camera,
		const mat23 &intrinsic,
		// The output
		GradientOfDenseImage &gradient,
		floatX<d_dense_image_residual_dim> &residual)
	{
		// PICP starts at 0
		computePointToPlanceICPTermJacobianResidual(
			target_vertex,
			target_normal,
			source_vertex,
			gradient,
			residual,
			0);
		// OpticalFlow starts at 1
		mat23 P;
		P.m00() = 1.f;
		P.m01() = 0.f;
		P.m02() = 0.f;
		P.m10() = 0.f;
		P.m11() = 1.f;
		P.m12() = 0.f;
		mat23 project_rot_world2camera = P * world2camera.rot;
		computeModifiedOpticalFlow2DTermJacobianResidual(
			target_vertex,
			source_vertex,
			project_rot_world2camera,
			// The output
			gradient,
			residual,
			1);
	}

	// This version needs to first check whether the given vertex will result in a match. If there is
	// a correspondence, the fill in the jacobian and residual values, else mark the value to zero.
	// The input is only depended on the SE(3) of the nodes, which can be updated without rebuild the index
	__global__ void computeDenseImageJacobianKernel(
		cudaTextureObject_t measure_vertex_confid_map,
		cudaTextureObject_t measure_normal_radius_map,
		cudaTextureObject_t reference_vertex_map,
		cudaTextureObject_t reference_normal_map,
		unsigned img_rows, unsigned img_cols,
		// The potential matched pixels and their knn
		GArrayView<ushort4> potential_matched_pixels, // (src, target)
		const unsigned short *__restrict__ potential_matched_knn_patch,
		const float *__restrict__ potential_matched_knn_patch_spatial_weight,
		const float *__restrict__ potential_matched_knn_patch_connect_weight,
		// The deformation
		const DualQuaternion *__restrict__ node_se3,
		const mat34 world2camera,
		const mat23 intrinsic,
		// The output
		GradientOfDenseImage *__restrict__ gradient,
		floatX<d_dense_image_residual_dim> *__restrict__ residual)
	{
		const auto idx = threadIdx.x + blockDim.x * blockIdx.x;
		if (idx >= potential_matched_pixels.Size())
			return;

		// 1. Prepare
		floatX<d_dense_image_residual_dim> pixel_residual;
		GradientOfDenseImage pixel_gradient;
		const ushort4 potential_pixel = potential_matched_pixels[idx];
		const unsigned short *knn_patch_ptr = potential_matched_knn_patch + idx * d_node_knn_size;
		const float *knn_patch_spatial_weight_ptr = potential_matched_knn_patch_spatial_weight + idx * d_node_knn_size;
		const float *knn_patch_connect_weight_ptr = potential_matched_knn_patch_connect_weight + idx * d_node_knn_size;

#ifdef OPT_DEBUG_CHECK
		bool flag_zero_average = false;
		DualQuaternion dq_average = averageDualQuaternionDebug(
			node_se3, knn_patch_ptr, knn_patch_spatial_weight_ptr, knn_patch_connect_weight_ptr, d_node_knn_size, flag_zero_average);
		if (flag_zero_average)
		{
			if (d_node_knn_size == 8)
			{
				printf("(%d, %d), nn: (%d, %d, %d, %d, %d, %d, %d, %d).\n",
					   potential_pixel.x, potential_pixel.y,
					   knn_patch_ptr[0], knn_patch_ptr[1], knn_patch_ptr[2], knn_patch_ptr[3],
					   knn_patch_ptr[4], knn_patch_ptr[5], knn_patch_ptr[6], knn_patch_ptr[7]);
			}
			else if (d_node_knn_size == 4) {
				printf("(%d, %d), nn: (%d, %d, %d, %d).\n",
					   potential_pixel.x, potential_pixel.y, knn_patch_ptr[0], knn_patch_ptr[1], knn_patch_ptr[2], knn_patch_ptr[3]);
			}
		}
#else
		DualQuaternion dq_average = averageDualQuaternion(
			node_se3, knn_patch_ptr, knn_patch_spatial_weight_ptr, knn_patch_connect_weight_ptr, d_node_knn_size);
#endif

		const mat34 se3 = dq_average.se3_matrix();

		// 1.2. Get the vertex
		const float4 can_vertex4 = tex2D<float4>(reference_vertex_map, potential_pixel.x, potential_pixel.y);
		const float4 can_normal4 = tex2D<float4>(reference_normal_map, potential_pixel.x, potential_pixel.y);

		// 1.3. Warp the vertex
		const float3 warped_vertex_world = se3.rot * can_vertex4 + se3.trans;
		const float3 warped_normal_world = se3.rot * can_normal4;

		// 1.4. Transfer to the camera frame
		const float3 warped_vertex_camera = world2camera.rot * warped_vertex_world + world2camera.trans;
		const float3 warped_normal_camera = world2camera.rot * warped_normal_world;

		// 1.5 Get target pixel
		ushort2 target_pixel = make_ushort2(potential_pixel.z, potential_pixel.w);

		// 2. Compute Jacobian term & Residual
		if (target_pixel.x >= 0 && target_pixel.x < img_cols && target_pixel.y >= 0 && target_pixel.y < img_rows)
		{
			// 2.1. Query the depth image
			const float4 target_vertex4 = tex2D<float4>(measure_vertex_confid_map, target_pixel.x, target_pixel.y);
			const float4 target_normal4 = tex2D<float4>(measure_normal_radius_map, target_pixel.x, target_pixel.y);
			const float3 target_vertex = make_float3(target_vertex4.x, target_vertex4.y, target_vertex4.z);
			const float3 target_normal = make_float3(target_normal4.x, target_normal4.y, target_normal4.z);

			// 2.2. Check the matched
			bool valid_pair = true;

			// 2.2.1. The depth pixel is not valid
			if (is_zero_vertex(target_vertex4))
			{
				valid_pair = false;
			}
			// 2.2.2. The orientation is not matched
			if (dot(target_normal, warped_normal_camera) < d_correspondence_normal_dot_threshold)
			{
				valid_pair = false;
			}
			// 2.2.3. The z-distance is too far away
			if (fabs(target_vertex.z - warped_vertex_camera.z) > d_correspondence_distance_threshold)
			{
				valid_pair = false;
			}

			// 2.2.4. This pair is valid, compute the jacobian and residual from two source: picp & opticalflow
			if (valid_pair)
			{
				float2 target_pixel_float = make_float2(target_pixel.x, target_pixel.y);
				float3 target_vertex_world = world2camera.apply_inversed_se3(target_vertex);
				float3 target_normal_world = world2camera.rot.transpose_dot(target_normal);
				computeDenseImageTermJacobianResidual(
					target_vertex_world,
					target_normal_world,
					target_pixel_float,
					warped_vertex_world,
					world2camera,
					intrinsic,
					// The output
					pixel_gradient,
					pixel_residual);
			}
		} // This pixel is projected inside

#ifdef OPT_DEBUG_CHECK
		for (auto c = 0; c < d_dense_image_residual_dim; ++c)
		{
			float *gradient_array = (float *)&pixel_gradient.gradient[c];
			for (auto i = 0; i < d_node_variable_dim; ++i)
			{
				if (gradient_array[i] > 1e3f || isnan(gradient_array[i]))
				{
					printf("Dense Jacobian Redisual: term: %d, channel: %d, jacobian: (%f, %f, %f, %f, %f, %f), can: (%f, %f, %f), warp: (%f, %f, %f).\n",
						   idx, c,
						   pixel_gradient.gradient[c].rotation.x,
						   pixel_gradient.gradient[c].rotation.y,
						   pixel_gradient.gradient[c].rotation.z,
						   pixel_gradient.gradient[c].translation.x,
						   pixel_gradient.gradient[c].translation.y,
						   pixel_gradient.gradient[c].translation.z,
						   can_vertex4.x, can_vertex4.y, can_vertex4.z,
						   warped_vertex_world.x, warped_vertex_world.y, warped_vertex_world.z);
				}
			}
		}
#endif // OPT_DEBUG_CHECK

		// 3. Write it to global memory
		residual[idx] = pixel_residual;
		gradient[idx] = pixel_gradient;
	}
}

/* The method and buffer for gradient computation
 */
void star::DenseImageHandler::ComputeJacobianTermsFixedIndex(cudaStream_t stream)
{
	for (auto cam_idx = 0; cam_idx < m_num_cam; ++cam_idx)
	{
		m_term_residual[cam_idx].ResizeArrayOrException(m_potential_pixels_knn.pixels[cam_idx].Size());
		m_term_gradient[cam_idx].ResizeArrayOrException(m_potential_pixels_knn.pixels[cam_idx].Size());

		dim3 blk(128);
		// dim3 blk(64);
		dim3 grid(divUp(m_potential_pixels_knn.pixels[cam_idx].Size(), blk.x));
		device::computeDenseImageJacobianKernel<<<grid, blk, 0, stream>>>(
			m_depth_observation.vertex_map[cam_idx],
			m_depth_observation.normal_map[cam_idx],
			m_geometry_maps.reference_vertex_map[cam_idx],
			m_geometry_maps.reference_normal_map[cam_idx],
			m_knn_map[cam_idx].Rows(), m_knn_map[cam_idx].Cols(),
			// The potential matched pixels and their knn
			m_potential_pixels_knn.pixels[cam_idx], // (src, target)
			m_potential_pixels_knn.surfel_knn_patch[cam_idx].Ptr(),
			m_potential_pixels_knn.knn_patch_spatial_weight[cam_idx].Ptr(),
			m_potential_pixels_knn.knn_patch_connect_weight[cam_idx].Ptr(), // Need to be build
			// The deformation
			m_node_se3.Ptr(),
			m_world2cam[cam_idx],
			mat23(m_project_intrinsic[cam_idx]),
			// The output
			m_term_gradient[cam_idx].Ptr(),
			m_term_residual[cam_idx].Ptr());
	}
	// Merge from all cameras
	// cudaSafeCall(cudaStreamSynchronize(stream));
	MergeTerm2Jacobian(stream);
	cudaSafeCall(cudaStreamSynchronize(stream));

#ifdef OPT_DEBUG_CHECK
	jacobianTermCheck();
#endif // OPT_DEBUG_CHECK

	// Sync and check error
#if defined(CUDA_DEBUG_SYNC_CHECK)
	cudaSafeCall(cudaStreamSynchronize(stream));
	cudaSafeCall(cudaGetLastError());
#endif
}

void star::DenseImageHandler::MergeTerm2Jacobian(cudaStream_t stream)
{
	unsigned offset = 0;
	for (auto cam_idx = 0; cam_idx < m_num_cam; ++cam_idx)
	{
		const auto num_pixels = m_term_residual[cam_idx].ArraySize();
		cudaSafeCall(cudaMemcpyAsync(
			m_term_residual_merge.Ptr() + offset,
			m_term_residual[cam_idx].Ptr(),
			m_term_residual[cam_idx].View().ByteSize(),
			cudaMemcpyDeviceToDevice,
			stream));
		cudaSafeCall(cudaMemcpyAsync(
			m_term_gradient_merge.Ptr() + offset,
			m_term_gradient[cam_idx].Ptr(),
			m_term_gradient[cam_idx].View().ByteSize(),
			cudaMemcpyDeviceToDevice,
			stream));
		offset += num_pixels;
	}
	m_term_residual_merge.ResizeArrayOrException(offset);
	m_term_gradient_merge.ResizeArrayOrException(offset);
}

star::DenseImageTerm2Jacobian star::DenseImageHandler::Term2JacobianMap() const
{
	DenseImageTerm2Jacobian term2jacobian;
	term2jacobian.knn_patch_array = m_potential_pixels_knn.surfel_knn_patch_all;
	term2jacobian.knn_patch_spatial_weight_array = m_potential_pixels_knn.knn_patch_spatial_weight_all;
	term2jacobian.knn_patch_connect_weight_array = m_potential_pixels_knn.knn_patch_connect_weight_all;
	term2jacobian.knn_patch_dq_array = m_potential_pixels_knn.knn_patch_dq_all;
	;
	term2jacobian.residual_array = m_term_residual_merge.View();
	term2jacobian.gradient_array = m_term_gradient_merge.View();
	term2jacobian.check_size();

	// Check correct
	return term2jacobian;
}