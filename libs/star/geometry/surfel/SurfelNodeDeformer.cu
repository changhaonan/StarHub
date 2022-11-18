#include <curand.h>
#include <curand_kernel.h>
#include <device_launch_parameters.h>
// Common
#include <star/common/types/typeX.h>
#include <star/common/macro_utils.h>
#include <star/math/DualQuaternion.hpp>
#include <star/geometry/surfel/SurfelNodeDeformer.h>

namespace star::device
{

	__global__ void ForwardWarpVertexAndNodeKernel(
		const float4 *__restrict__ reference_vertex_confid,
		const float4 *__restrict__ reference_normal_radius,
		const ushortX<d_surfel_knn_size> *__restrict__ vertex_knn_array,
		const floatX<d_surfel_knn_size> *__restrict__ vertex_knn_spatial_weight,
		const floatX<d_surfel_knn_size> *__restrict__ vertex_knn_connect_weight,
		// For nodes
		const float4 *__restrict__ reference_node_coordinate,
		const ushortX<d_node_knn_size> *__restrict__ node_knn_array,
		const floatX<d_node_knn_size> *__restrict__ node_knn_spatial_weight,
		const floatX<d_node_knn_size> *__restrict__ node_knn_connect_weight,
		const DualQuaternion *__restrict__ warp_field,
		// Output array, shall be size correct
		float4 *__restrict__ live_vertex_confid,
		float4 *__restrict__ live_normal_radius,
		float4 *__restrict__ live_node_coordinate,
		const unsigned vertex_size,
		const unsigned node_size)
	{
		const int idx = threadIdx.x + blockDim.x * blockIdx.x;
		float4 vertex;
		float4 normal;
		DualQuaternion dq_average;

		if (idx < vertex_size)
		{
			// Warp Surfel
			auto knn = vertex_knn_array[idx];
			auto spatial_weight = vertex_knn_spatial_weight[idx];
			auto connect_weight = vertex_knn_connect_weight[idx];

			vertex = reference_vertex_confid[idx];
			normal = reference_normal_radius[idx];
			dq_average = averageDualQuaternion<d_node_knn_size>(
				warp_field, knn, spatial_weight, connect_weight);
		}
		else if (idx >= vertex_size && idx < vertex_size + node_size)
		{
			// Warp Node
			const int offset = idx - vertex_size;
			auto knn = node_knn_array[offset];
			auto spatial_weight = node_knn_spatial_weight[offset];
			auto connect_weight = node_knn_connect_weight[offset];
			vertex = reference_node_coordinate[offset];
			dq_average = averageDualQuaternion<d_node_knn_size>(
				warp_field, knn, spatial_weight, connect_weight);
		}

		const mat34 se3 = dq_average.se3_matrix();
		float3 v3 = make_float3(vertex.x, vertex.y, vertex.z);
		float3 n3 = make_float3(normal.x, normal.y, normal.z);
		v3 = se3.rot * v3 + se3.trans;
		n3 = se3.rot * n3;
		vertex = make_float4(v3.x, v3.y, v3.z, vertex.w);
		normal = make_float4(n3.x, n3.y, n3.z, normal.w);

		// Save it
		if (idx < vertex_size)
		{
			live_vertex_confid[idx] = vertex;
			live_normal_radius[idx] = normal;
		}
		else if (idx >= vertex_size && idx < vertex_size + node_size)
		{
			const int offset = idx - vertex_size;
			live_node_coordinate[offset] = vertex;
		}
	}

	__global__ void InverseWarpVertexNormalKernel(
		const GArrayView<float4> live_vertex_confid_array,
		const float4 *live_normal_radius_array,
		const ushort4 *vertex_knn_array,
		const float4 *vertex_knn_weight,
		const DualQuaternion *device_warp_field,
		float4 *reference_vertex_confid,
		float4 *reference_normal_radius)
	{
		const int idx = threadIdx.x + blockDim.x * blockIdx.x;
		if (idx < live_vertex_confid_array.Size())
		{
			const float4 live_vertex_confid = live_vertex_confid_array[idx];
			const float4 live_normal_radius = live_normal_radius_array[idx];
			const ushort4 knn = vertex_knn_array[idx];
			const float4 knn_weight = vertex_knn_weight[idx];
			auto dq_average = averageDualQuaternion(device_warp_field, knn, knn_weight);
			mat34 se3 = dq_average.se3_matrix();
			float3 vertex = make_float3(live_vertex_confid.x, live_vertex_confid.y, live_vertex_confid.z);
			float3 normal = make_float3(live_normal_radius.x, live_normal_radius.y, live_normal_radius.z);
			// Apply the inversed warping without construction of the matrix
			vertex = se3.apply_inversed_se3(vertex);
			normal = se3.rot.transpose_dot(normal);
			reference_vertex_confid[idx] = make_float4(vertex.x, vertex.y, vertex.z, live_vertex_confid.w);
			reference_normal_radius[idx] = make_float4(normal.x, normal.y, normal.z, live_normal_radius.w);
		}
	}

	__global__ void SetupRandomKernel(curandState *state)
	{
		const int idx = threadIdx.x + blockDim.x * blockIdx.x;
		curand_init(1234, idx, 0, &state[idx]);
	}

	__global__ void GenerateRandomDeformationKernel(
		curandState *state,
		DualQuaternion *__restrict__ node_se3,
		const float trans_scale,
		const float rot_scale)
	{
		const int idx = threadIdx.x + blockDim.x * blockIdx.x;

		const float trans_x = curand_uniform(state + 4 * idx) * trans_scale;
		const float trans_y = curand_uniform(state + 4 * idx + 1) * trans_scale;
		const float trans_z = curand_uniform(state + 4 * idx + 2) * trans_scale;
		const float3 twist_trans = make_float3(trans_x, trans_y, trans_z);

		const float rot_x = curand_uniform(state + 4 * idx + 3) * rot_scale;
		const float rot_y = curand_uniform(state + 4 * idx + 4) * rot_scale;
		const float rot_z = curand_uniform(state + 4 * idx + 5) * rot_scale;
		const float3 twist_rot = make_float3(rot_x, rot_y, rot_z);

		DualQuaternion dq;
		dq.set_identity();
		apply_twist(twist_rot, twist_trans, dq);

		node_se3[idx] = dq;
	}
}

void star::SurfelNodeDeformer::ForwardWarpSurfelsAndNodes(
	WarpField::DeformationAcess warp_field,
	SurfelGeometry &geometry,
	const GArrayView<DualQuaternion> &node_se3,
	cudaStream_t stream)
{
	// Check the size
	CheckSurfelGeometySize(geometry);

	// The node se3 should have the same size
	STAR_CHECK_EQ(node_se3.Size(), warp_field.reference_node_coords.Size());

	// Invoke kernel
	dim3 blk(256);
	dim3 grid(divUp(geometry.NumValidSurfels() + warp_field.reference_node_coords.Size(), blk.x));
	device::ForwardWarpVertexAndNodeKernel<<<grid, blk, 0, stream>>>(
		geometry.ReferenceVertexConfidenceReadOnly().Ptr(),
		geometry.ReferenceNormalRadiusReadOnly().Ptr(),
		geometry.SurfelKNNReadOnly().Ptr(),
		geometry.SurfelKNNSpatialWeightReadOnly().Ptr(),
		geometry.SurfelKNNConnectWeightReadOnly().Ptr(),
		// For nodes
		warp_field.reference_node_coords.Ptr(),
		warp_field.node_knn.Ptr(),
		warp_field.node_knn_spatial_weight.Ptr(),
		warp_field.node_knn_connect_weight.Ptr(),
		node_se3.Ptr(),
		// Output array, shall be size correct
		geometry.LiveVertexConfidence().Ptr(),
		geometry.LiveNormalRadius().Ptr(),
		warp_field.live_node_coords.Ptr(),
		geometry.NumValidSurfels(),
		warp_field.reference_node_coords.Size());

	// Sync and check error
#if defined(CUDA_DEBUG_SYNC_CHECK)
	cudaSafeCall(cudaStreamSynchronize(stream));
	cudaSafeCall(cudaGetLastError());
#endif
}

void star::SurfelNodeDeformer::GenerateRandomDeformation(
	GArraySlice<DualQuaternion> &node_se3,
	const float trans_scale,
	const float rot_scale,
	cudaStream_t stream)
{
	// Set random state
	curandState *d_state;
	cudaSafeCall(cudaMallocAsync(&d_state, sizeof(curandState) * node_se3.Size(), stream));
	dim3 blk(128);
	dim3 grid = dim3(divUp(6 * node_se3.Size(), blk.x));
	device::SetupRandomKernel<<<grid, blk, 0, stream>>>(d_state);

	// Generate random DualQuaternion
	grid = dim3(divUp(node_se3.Size(), blk.x));
	device::GenerateRandomDeformationKernel<<<grid, blk, 0, stream>>>(
		d_state,
		node_se3.Ptr(),
		trans_scale,
		rot_scale);

	cudaSafeCall(cudaFreeAsync(d_state, stream));
	cudaSafeCall(cudaStreamSynchronize(stream));
}

void star::SurfelNodeDeformer::CheckSurfelGeometySize(const SurfelGeometry &geometry)
{
	const auto num_surfels = geometry.NumValidSurfels();
	STAR_CHECK(geometry.m_reference_vertex_confid.ArraySize() == num_surfels);
	STAR_CHECK(geometry.m_reference_normal_radius.ArraySize() == num_surfels);
	STAR_CHECK(geometry.m_surfel_knn.ArraySize() == num_surfels);
	STAR_CHECK(geometry.m_surfel_knn_spatial_weight.ArraySize() == num_surfels);
	STAR_CHECK(geometry.m_live_vertex_confid.ArraySize() == num_surfels);
	STAR_CHECK(geometry.m_live_normal_radius.ArraySize() == num_surfels);
}