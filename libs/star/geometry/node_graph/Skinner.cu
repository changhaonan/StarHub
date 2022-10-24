#include "star/common/Constants.h"
#include "star/geometry/node_graph/Skinner.h"
#include "star/geometry/node_graph/brute_force_knn.cuh"
#include "math/vector_ops.hpp"
#include <device_launch_parameters.h>


namespace star { namespace device {

	__global__ void SkinnerKernel(
		const float4* __restrict__ node_coordinate,
		const uint2* __restrict__ node_status,
		const float4* __restrict__ vertex_confid,
		ushortX<d_surfel_knn_size>* __restrict__ vertex_knn,
		floatX<d_surfel_knn_size>* __restrict__ vertex_knn_spatial_weight,
		floatX<d_surfel_knn_size>* __restrict__ vertex_knn_connect_weight,
		const unsigned vertex_size,
		const unsigned node_size,
		const float node_radius_square
	) {
		const auto idx = threadIdx.x + blockIdx.x * blockDim.x;
		if (idx >= vertex_size) return;

		// 1. Initialization
		float4 v = vertex_confid[idx];
		float distance[d_surfel_knn_size];
		unsigned short index[d_surfel_knn_size];
#pragma unroll
		for (auto i = 0; i < d_surfel_knn_size; ++i) {
			distance[i] = 1e6f;
			index[i] = 0;
		}
		KnnHeapExpandDevice<d_surfel_knn_size> knn_heap(distance, index);
		
		// 2. Traverse all existing nodes
		for (auto i = 0; i < node_size; ++i) {
			float dist = 1e6f;
			if (node_status[i].y != DELETED_NODE_STATUS) {
				float4 v_node = node_coordinate[i];
				float dx = (v.x - v_node.x);
				float dy = (v.y - v_node.y);
				float dz = (v.z - v_node.z);
				dist = dx * dx + dy * dy + dz * dz;
			}
			knn_heap.update(i, dist);
		}
		knn_heap.sort();

		float weight[d_surfel_knn_size];
		float weight_sum = 0.f;
		const float3 v3 = make_float3(v.x, v.y, v.z);
#pragma unroll
		for (auto i = 0; i < d_surfel_knn_size; ++i) {
			const float4 node_v = node_coordinate[knn_heap.index[i]];
			const float3 node_v3 = make_float3(node_v.x, node_v.y, node_v.z);
			const float dist = squared_norm(v3 - node_v3);
			weight[i] = __expf(-dist / (2.f * node_radius_square));
			weight_sum += weight[i];
		}
#pragma unroll
		float weight_sum_inv = 1.f / weight_sum;
		for (auto i = 0; i < d_surfel_knn_size; ++i) {
			vertex_knn[idx][i] = knn_heap.index[i];
#if defined(USE_INTERPOLATE_WEIGHT_NORMALIZATION)
			vertex_knn_spatial_weight[idx][i] = weight_sum_inv * weight[i];
#else
			vertex_knn_spatial_weight[idx][i] = weight[i];
#endif

#ifdef OPT_DEBUG_CHECK
			if (isnan(vertex_knn_spatial_weight[idx][i])) {
				const float4 node_v = node_coordinate[knn_heap.index[i]];
				const float3 node_v3 = make_float3(node_v.x, node_v.y, node_v.z);
				const float dist = squared_norm(v3 - node_v3);
				printf("surfel: %d, v3: (%f, %f, %f), v_node: (%f, %f, %f), dist: %f, weight_sum: %f, weight: %f\n",
					idx, v3.x, v3.y, v3.z, node_v3.x, node_v3.y, node_v3.z, dist, weight_sum, weight[i]);
			}
#endif
		}

		// 3. Connect weight
#pragma unroll
		for (auto i = 0; i < d_surfel_knn_size; ++i) {
			vertex_knn_connect_weight[idx][i] = 1.f;  // Fix it for now
		}
	}


	__global__ void SkinnerKernelWithSemantic(
		const float4* __restrict__ node_coordinate,
		const uint2* __restrict__ node_status,
		const float4* __restrict__ vertex_confid,
		const ucharX<d_max_num_semantic>* __restrict__ node_semantic_prob,
		const ucharX<d_max_num_semantic>* __restrict__ surfel_semantic_prob,
		ushortX<d_surfel_knn_size>* __restrict__ vertex_knn,
		floatX<d_surfel_knn_size>* __restrict__ vertex_knn_spatial_weight,
		floatX<d_surfel_knn_size>* __restrict__ vertex_knn_connect_weight,
		const unsigned vertex_size,
		const unsigned node_size,
		const float node_radius_square
	) {
		const auto idx = threadIdx.x + blockIdx.x * blockDim.x;
		if (idx >= vertex_size) return;

		// 1. Initialization
		float4 v = vertex_confid[idx];
		float distance[d_surfel_knn_size];
		unsigned short index[d_surfel_knn_size];
#pragma unroll
		for (auto i = 0; i < d_surfel_knn_size; ++i) {
			distance[i] = 1e6f;
			index[i] = 0;
		}
		KnnHeapExpandDevice<d_surfel_knn_size> knn_heap(distance, index);

		// 2. Traverse all existing nodes
		for (auto i = 0; i < node_size; ++i) {
			float dist = 1e6f;
			// 2.1. Check node_status & node_semantic
			if (node_status[i].y != DELETED_NODE_STATUS) {
				float4 v_node = node_coordinate[i];
				float dx = (v.x - v_node.x);
				float dy = (v.y - v_node.y);
				float dz = (v.z - v_node.z);
				dist = dx * dx + dy * dy + dz * dz;
			}
			knn_heap.update(i, dist);
		}
		knn_heap.sort();

		float weight[d_surfel_knn_size];
		float weight_sum = 0.f;
		const float3 v3 = make_float3(v.x, v.y, v.z);
#pragma unroll
		for (auto i = 0; i < d_surfel_knn_size; ++i) {
			const float4 node_v = node_coordinate[knn_heap.index[i]];
			const float3 node_v3 = make_float3(node_v.x, node_v.y, node_v.z);
			const float dist = squared_norm(v3 - node_v3);
			weight[i] = __expf(-dist / (2.f * node_radius_square));
			weight_sum += weight[i];
		}
#pragma unroll
		float weight_sum_inv = 1.f / weight_sum;
		for (auto i = 0; i < d_surfel_knn_size; ++i) {
			vertex_knn[idx][i] = knn_heap.index[i];
#if defined(USE_INTERPOLATE_WEIGHT_NORMALIZATION)
			vertex_knn_spatial_weight[idx][i] = weight_sum_inv * weight[i];
#else
			vertex_knn_spatial_weight[idx][i] = weight[i];
#endif

#ifdef OPT_DEBUG_CHECK
			if (isnan(vertex_knn_spatial_weight[idx][i])) {
				const float4 node_v = node_coordinate[knn_heap.index[i]];
				const float3 node_v3 = make_float3(node_v.x, node_v.y, node_v.z);
				const float dist = squared_norm(v3 - node_v3);
				printf("surfel: %d, v3: (%f, %f, %f), v_node: (%f, %f, %f), dist: %f, weight_sum: %f, weight: %f\n",
					idx, v3.x, v3.y, v3.z, node_v3.x, node_v3.y, node_v3.z, dist, weight_sum, weight[i]);
			}
#endif
		}

		// 3. Connect weight
		int surfel_label = max_id(surfel_semantic_prob[idx]);
#pragma unroll
		for (auto i = 0; i < d_surfel_knn_size; ++i) {
			auto node_idx = knn_heap.index[i];
			int node_label = max_id(node_semantic_prob[node_idx]);
			if (surfel_label != node_label) {
				vertex_knn_connect_weight[idx][i] = 1e-12f;  // Suppress connection weight from different semantic, can't set to zero due to numerical stablity
			}
			else {
				vertex_knn_connect_weight[idx][i] = 1.f;  // Set it to full connection
			}
		}

		// 4. Debug
#ifdef OPT_DEBUG_CHECK
		bool flag_zero_connect = true;
		for (auto i = 0; i < d_surfel_knn_size; ++i) {
			if (vertex_knn_connect_weight[idx][i] != 0.f) {
				flag_zero_connect = false;
			}
		}
		if (flag_zero_connect) {
			printf("Skinning: Zero connect at surfel %d.\n", index);
		}
#endif // OPT_DEBUG_CHECK

	}


	__global__ void IncSkinnerKernel(
		const float4* __restrict__ node_coordinate,
		const uint2* __restrict__ node_status,
		const float4* __restrict__ vertex_confid,
		ushortX<d_surfel_knn_size>* __restrict__ vertex_knn,
		floatX<d_surfel_knn_size>* __restrict__ vertex_knn_spatial_weight,
		floatX<d_surfel_knn_size>* __restrict__ vertex_knn_connect_weight,
		const unsigned vertex_size,
		const unsigned node_size,
		const float node_radius_square,
		const unsigned num_prev_node,
		const unsigned num_remaining_surfel
	) {
		const auto idx = threadIdx.x + blockIdx.x * blockDim.x;
		if (idx >= (vertex_size - num_remaining_surfel)) return;

		// 1. Initialization with previous results
		float4 v = vertex_confid[num_remaining_surfel + idx];
		const float3 v3 = make_float3(v.x, v.y, v.z);
		float distance[d_surfel_knn_size];
		unsigned short index[d_surfel_knn_size];
		auto prev_knn = vertex_knn[num_remaining_surfel + idx];
#pragma unroll
		for (auto i = 0; i < d_surfel_knn_size; ++i) {
			index[i] = prev_knn[i];
			const float4 node_v = node_coordinate[prev_knn[i]];
			const float3 node_v3 = make_float3(node_v.x, node_v.y, node_v.z);
			const float dist = squared_norm(v3 - node_v3);
			distance[i] = dist;
		}
		KnnHeapExpandDevice<d_surfel_knn_size> knn_heap(distance, index);

		// 2. Traverse all newly appended nodes
		for (auto i = num_prev_node; i < node_size; ++i) {
			float dist = 1e6f;
			if (node_status[i].y != DELETED_NODE_STATUS) {
				float4 v_node = node_coordinate[i];
				float dx = (v.x - v_node.x);
				float dy = (v.y - v_node.y);
				float dz = (v.z - v_node.z);
				dist = dx * dx + dy * dy + dz * dz;
			}
			knn_heap.update(i, dist);
		}
		knn_heap.sort();

		// 3. Average setting weight
		float weight[d_surfel_knn_size];
		float weight_sum = 0.f;
#pragma unroll
		for (auto i = 0; i < d_surfel_knn_size; ++i) {
			const float4 node_v = node_coordinate[knn_heap.index[i]];
			const float3 node_v3 = make_float3(node_v.x, node_v.y, node_v.z);
			const float dist = squared_norm(v3 - node_v3);
			weight[i] = __expf(-dist / (2.f * node_radius_square));
			weight_sum += weight[i];
		}
#pragma unroll
		float weight_sum_inv = 1.f / weight_sum;
		for (auto i = 0; i < d_surfel_knn_size; ++i) {
			vertex_knn[num_remaining_surfel + idx][i] = knn_heap.index[i];
#if defined(USE_INTERPOLATE_WEIGHT_NORMALIZATION)
			vertex_knn_spatial_weight[num_remaining_surfel + idx][i] = weight_sum_inv * weight[i];
#else
			vertex_knn_spatial_weight[num_remaining_surfel + idx][i] = weight[i];
#endif

#ifdef OPT_DEBUG_CHECK
			if (isnan(vertex_knn_spatial_weight[num_remaining_surfel + idx][i])) {
				const float4 node_v = node_coordinate[knn_heap.index[i]];
				const float3 node_v3 = make_float3(node_v.x, node_v.y, node_v.z);
				const float dist = squared_norm(v3 - node_v3);
				printf("surfel: %d, v3: (%f, %f, %f), v_node: (%f, %f, %f), dist: %f, weight_sum: %f, weight: %f\n",
					num_remaining_surfel + idx, v3.x, v3.y, v3.z, node_v3.x, node_v3.y, node_v3.z, dist, weight_sum, weight[i]);
			}
#endif
		}

		// 4. Set Connect weight
#pragma unroll
		for (auto i = 0; i < d_surfel_knn_size; ++i) {
			vertex_knn_connect_weight[num_remaining_surfel + idx][i] = 1.f;  // Fix it for now
		}
	}


	__global__ void UpdateSkinningConnectionKernel(
		const ucharX<d_max_num_semantic>* __restrict__ node_semantic_prob,
		const ucharX<d_max_num_semantic>* __restrict__ surfel_semantic_prob,
		const ushortX<d_surfel_knn_size>* __restrict__ vertex_knn,
		floatX<d_surfel_knn_size>* __restrict__ vertex_knn_connect_weight,
		const unsigned vertex_size) {

		const auto idx = threadIdx.x + blockIdx.x * blockDim.x;
		if (idx >= vertex_size) return;

		int surfel_label = max_id(surfel_semantic_prob[idx]);
		auto knn = vertex_knn[idx];
#pragma unroll
		for (auto i = 0; i < d_surfel_knn_size; ++i) {
			auto node_idx = knn[i];
			int node_label = max_id(node_semantic_prob[node_idx]);
			if (surfel_label != node_label) {
				vertex_knn_connect_weight[idx][i] = 1e-12f;  // Suppress connection weight from different semantic, can't set to zero due to numerical stablity
			}
			else {
				vertex_knn_connect_weight[idx][i] = 1.f;  // Set it to full connection
			}
		}
	}

}
}

void star::Skinner::PerformSkinningFromRef(
	Geometry4Skinner& geometry, NodeGraph4Skinner& node_graph, cudaStream_t stream) {
	const auto vertex_size = geometry.reference_vertex_confid.Size();
	const auto node_size = node_graph.node_status.Size();
	dim3 blk(128);
	dim3 grid(divUp(vertex_size, blk.x));
	device::SkinnerKernel<<<grid, blk, 0, stream>>>(
		node_graph.reference_node_coords.Ptr(),
		node_graph.node_status.Ptr(),
		geometry.reference_vertex_confid.Ptr(),
		geometry.surfel_knn.Ptr(),
		geometry.surfel_knn_spatial_weight.Ptr(),
		geometry.surfel_knn_connect_weight.Ptr(),
		vertex_size,
		node_size,
		node_graph.node_radius_square
	);
}

void star::Skinner::PerformSkinningFromLive(
	Geometry4Skinner& geometry, NodeGraph4Skinner& node_graph, cudaStream_t stream) {
	const auto vertex_size = geometry.reference_vertex_confid.Size();
	const auto node_size = node_graph.node_status.Size();
	dim3 blk(128);
	dim3 grid(divUp(vertex_size, blk.x));
	device::SkinnerKernel<<<grid, blk, 0, stream>>>(
		node_graph.live_node_coords.Ptr(),
		node_graph.node_status.Ptr(),
		geometry.live_vertex_confid.Ptr(),
		geometry.surfel_knn.Ptr(),
		geometry.surfel_knn_spatial_weight.Ptr(),
		geometry.surfel_knn_connect_weight.Ptr(),
		vertex_size,
		node_size,
		node_graph.node_radius_square);
}

void star::Skinner::PerformSkinningFromLiveWithSemantic(
	Geometry4Skinner& geometry, NodeGraph4Skinner& node_graph, cudaStream_t stream) {
	const auto vertex_size = geometry.reference_vertex_confid.Size();
	const auto node_size = node_graph.node_status.Size();
	dim3 blk(128);
	dim3 grid(divUp(vertex_size, blk.x));
	device::SkinnerKernelWithSemantic<<<grid, blk, 0, stream>>>(
		node_graph.live_node_coords.Ptr(),
		node_graph.node_status.Ptr(),
		geometry.live_vertex_confid.Ptr(),
		node_graph.node_semantic_prob.Ptr(),
		geometry.surfel_semantic_prob.Ptr(),
		geometry.surfel_knn.Ptr(),
		geometry.surfel_knn_spatial_weight.Ptr(),
		geometry.surfel_knn_connect_weight.Ptr(),
		vertex_size,
		node_size,
		node_graph.node_radius_square);
}

void star::Skinner::PerformIncSkinnningFromLive(
	Geometry4Skinner& geometry, NodeGraph4Skinner& node_graph, 
	const unsigned num_prev_node_size, const unsigned num_remaining_surfel, cudaStream_t stream
) {
	const auto vertex_size = geometry.reference_vertex_confid.Size();
	const auto node_size = node_graph.node_status.Size();
	dim3 blk(128);
	dim3 grid(divUp(vertex_size, blk.x));
	device::IncSkinnerKernel<<<grid, blk, 0, stream>>>(
		node_graph.live_node_coords.Ptr(),
		node_graph.node_status.Ptr(),
		geometry.live_vertex_confid.Ptr(),
		geometry.surfel_knn.Ptr(),
		geometry.surfel_knn_spatial_weight.Ptr(),
		geometry.surfel_knn_connect_weight.Ptr(),
		vertex_size,
		node_size,
		node_graph.node_radius_square,
		num_prev_node_size,
		num_remaining_surfel);
}

void star::Skinner::UpdateSkinnningConnection(
	Geometry4Skinner& geometry, NodeGraph4Skinner& node_graph, cudaStream_t stream
) {
	const auto vertex_size = geometry.reference_vertex_confid.Size();
	dim3 blk(128);
	dim3 grid(divUp(vertex_size, blk.x));

	device::UpdateSkinningConnectionKernel<<<grid, blk, 0, stream>>>(
		node_graph.node_semantic_prob.Ptr(),
		geometry.surfel_semantic_prob.Ptr(),
		geometry.surfel_knn.Ptr(),
		geometry.surfel_knn_connect_weight.Ptr(),
		vertex_size);
}
