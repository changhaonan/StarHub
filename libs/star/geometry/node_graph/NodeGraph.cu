#include "star/common/Constants.h"
#include "star/geometry/node_graph/NodeGraph.h"
#include "math/vector_ops.hpp"
#include "star/geometry/node_graph/brute_force_knn.cuh"
#include <device_launch_parameters.h>


namespace star { namespace device {
	
	__global__ void UpdateNodeDistanceKernel(
		const float4* __restrict__ reference_node_coordinate,
		const uint2* __restrict__ node_status,
		half* __restrict__ node_distance,
		const int node_size
	) {
		const auto idx = threadIdx.x + blockIdx.x * blockDim.x;
		const auto idy = threadIdx.y + blockIdx.y * blockDim.y;
		if (idx >= node_size || idy >= node_size || idx < idy) return;
		// Check status first
		half max_distance_history = 1e6f;

		// Compute distance
		if (node_status[idx].y != DELETED_NODE_STATUS && node_status[idy].y != DELETED_NODE_STATUS) {
			const float4 v1 = reference_node_coordinate[idx];
			const float4 v2 = reference_node_coordinate[idy];
			const float dx = v1.x - v2.x;
			const float dy = v1.y - v2.y;
			const float dz = v1.z - v2.z;

			const float dx_sq = __fmul_rn(dx, dx);
			const float dxy_sq = __fmaf_rn(dy, dy, dx_sq);
			const float dist = __fmaf_rn(dz, dz, dxy_sq);
			max_distance_history = __float2half_rn(dist);
		}
		
		// Update current distance
		const auto id_xy = idx * d_max_num_nodes + idy;
		const auto id_yx = idy * d_max_num_nodes + idx;
		node_distance[id_xy] = max_distance_history;
		node_distance[id_yx] = max_distance_history;
	}

	
	/**
	* Build node graph based on node distance
	*/
	__global__ void BuildNodeGraphKernel(
		const half* __restrict__ distance,
		ushortX<d_node_knn_size>* __restrict__ node_knn,
		floatX<d_node_knn_size>* __restrict__ node_knn_spatial_weight,
		const unsigned node_size,
		const float node_radius_square
	) {
		const auto idx = threadIdx.x + blockIdx.x * blockDim.x;
		if (idx >= node_size) return;

		float nn_distance[d_node_knn_size];
		unsigned short nn_id[d_node_knn_size];
#pragma unroll
		for (unsigned i = 0; i < d_node_knn_size; ++i) {
			nn_distance[i] = 1e3f;
			nn_id[i] = 0;
		}

		KnnHeapExpandDevice<d_node_knn_size> knn_heap(nn_distance, nn_id);
		for (auto i = 0; i < node_size; ++i) {
			knn_heap.update(i, __half2float(distance[idx * d_max_num_nodes + i]));
		}
		knn_heap.sort();

		// Update nn idx
#pragma unroll
		for (unsigned i = 0; i < d_node_knn_size; ++i) {
			node_knn[idx][i] = knn_heap.index[i];
		}
		// Update spatial weight
#pragma unroll
		for (unsigned i = 0; i < d_node_knn_size; ++i) {
			node_knn_spatial_weight[idx][i] = __expf(-knn_heap.distance[i] / (2 * node_radius_square));
		}
#if defined(USE_INTERPOLATE_WEIGHT_NORMALIZATION)
		float weight_sum = 0.f;
#pragma unroll
		for (unsigned i = 0; i < d_node_knn_size; ++i) {
			weight_sum += node_knn_spatial_weight[idx][i];
		}
		const float inv_weight_sum = 1.0f / weight_sum;
#pragma unroll
		for (unsigned i = 0; i < d_node_knn_size; ++i) {
			node_knn_spatial_weight[idx][i] *= inv_weight_sum;
		}
#endif
	}
	
	
	__global__ void BuildNodeGraphPairKernel(
		const ushortX<d_node_knn_size>* __restrict__ node_knn,
		const uint2* __restrict__ node_status,
		ushort3* __restrict__ node_graph_pair,
		const unsigned node_size
	) {
		const auto idx = threadIdx.x + blockIdx.x * blockDim.x;
		if (idx >= node_size) return;
		const auto offset = d_node_knn_size * idx;
#pragma unroll
		for (auto i = 0; i < d_node_knn_size; ++i) {
			node_graph_pair[offset + i] = make_ushort3(idx, node_knn[idx][i], i);
		}
	}


	__global__ void UpdateAppendNodeInitialTimeKernel(
		unsigned* __restrict__ node_inital_time,
		const unsigned current_time,
		const unsigned prev_node_size,
		const unsigned append_node_size
	) {
		const auto idx = threadIdx.x + blockIdx.x * blockDim.x;
		if (idx >= append_node_size) return;
		node_inital_time[prev_node_size + idx] = current_time;
	}


	__global__ void ResetNodeConnectionKernel(
		floatX<d_node_knn_size>* __restrict__ node_knn_connect_weight,
		const unsigned node_size
	) {
		const auto idx = threadIdx.x + blockIdx.x * blockDim.x;
		if (idx >= node_size) return;

#pragma unroll
		for (auto i = 0; i < d_node_knn_size; ++i) {
			node_knn_connect_weight[idx][i] = 1.f;  // Reset as full connection
		}
	}

	__global__ void ComputeNodeGraphConnectionFromSemanticKernel(
		floatX<d_node_knn_size>* __restrict__ node_knn_connect_weight,
		const ushortX<d_node_knn_size>* __restrict__ node_knn,
		const ucharX<d_max_num_semantic>* __restrict__ node_semantic_prob,
		const floatX<d_max_num_semantic> dynamic_regulation,
		const unsigned node_size
	) {
		const auto idx = threadIdx.x + blockIdx.x * blockDim.x;
		if (idx >= node_size) return;

		// 1. Compare the semantic with neighbor
		int node_label = max_id(node_semantic_prob[idx]);
		for (auto i = 0; i < d_node_knn_size; ++i) {
			auto nn_idx = node_knn[idx][i];
			int nn_node_label = max_id(node_semantic_prob[nn_idx]);
			if (node_label != nn_node_label) {
				node_knn_connect_weight[idx][i] = 1e-12f;  // Make it supper small
			}
			else {
				// Apply different regulation to different type of objects
				node_knn_connect_weight[idx][i] = dynamic_regulation[node_label];
			}
		}
	}

}
}


void star::NodeGraph::InitializeNodeGraphFromNode(
	const GArrayView<float4>& node_coords,
    const unsigned current_time,
	const bool use_ref,
	cudaStream_t stream
) {
    m_node_size = 0;  // Set size to 0 first, because we are doing intialization
	updateNodeCoordinate(node_coords, current_time, stream);
	BuildNodeGraphFromScratch(use_ref, stream);  // Use ref & live are the same here
}

void star::NodeGraph::InitializeNodeGraphFromNode(
	const std::vector<float4>& node_coords,
	const unsigned current_time,
	const bool use_ref,
	cudaStream_t stream
) {
	m_node_size = 0;  // Set size to 0 first, because we are doing intialization
	updateNodeCoordinate(node_coords, current_time, stream);
	BuildNodeGraphFromScratch(use_ref, stream);  // Use ref & live are the same here
}

void star::NodeGraph::updateNodeCoordinate(
	const GArrayView<float4>& node_coords, const unsigned current_time, cudaStream_t stream) {
	cudaSafeCall(
		cudaMemcpyAsync(
			m_reference_node_coords.DevicePtr(),
			node_coords.Ptr(),
			node_coords.ByteSize(),
			cudaMemcpyDeviceToDevice,
			stream
		)
	);
	// If the new nodes is more than previous nodes
	assert(node_coords.Size() > m_node_size);  // Node graph size can only increase
	if (node_coords.Size() != m_node_size) {
        updateAppendNodeInitialTime(current_time, m_node_size, node_coords.Size() - m_node_size, stream);
    }
	// Sync on device
	cudaSafeCall(cudaStreamSynchronize(stream));
	resizeNodeSize(node_coords.Size());
}

void star::NodeGraph::updateNodeCoordinate(
	const std::vector<float4>& node_coords, const unsigned current_time, cudaStream_t stream) {
	// If the new nodes is more than previous nodes
	assert(node_coords.size() > m_node_size);  // Node graph size can only increase
	if (node_coords.size() != m_node_size) {
		updateAppendNodeInitialTime(current_time, m_node_size, node_coords.size() - m_node_size, stream);
	}
	// Check the size
	STAR_CHECK(node_coords.size() <= Constants::kMaxNumNodes) << "Too many nodes";
	// Copy the value to device
	cudaSafeCall(cudaMemcpyAsync(
		m_reference_node_coords.DevicePtr(),
		node_coords.data(),
		node_coords.size() * sizeof(float4),
		cudaMemcpyHostToDevice,
		stream
	));
	// Update size; Resize is only called here.
	cudaSafeCall(cudaStreamSynchronize(stream));
	resizeNodeSize(node_coords.size());
}

void star::NodeGraph::buildNodeGraphPair(cudaStream_t stream) {
	// Clear first
	cudaSafeCall(cudaMemsetAsync((void*)m_node_graph_pair.Ptr(), 0, sizeof(ushort3) * d_node_knn_size * m_node_size), stream);

	dim3 blk(128);
	dim3 grid(divUp(m_node_size, blk.x));
	device::BuildNodeGraphPairKernel<<<grid, blk, 0, stream>>>(
		m_node_knn.DevicePtr(),
		m_node_status.Ptr(),
		m_node_graph_pair.Ptr(),
		m_node_size
	);

	// Sync and check error
#if defined(CUDA_DEBUG_SYNC_CHECK)
	cudaSafeCall(cudaStreamSynchronize(stream));
	cudaSafeCall(cudaGetLastError());
#endif

#ifdef DEBUG_NODE_GRAPH
	CheckNodeGraphPairConsistency(stream);
#endif
}

void star::NodeGraph::UpdateNodeDistance(const bool use_ref, cudaStream_t stream) {
	dim3 blk(16, 16);
	dim3 grid(divUp(m_node_size, blk.x), divUp(m_node_size, blk.y));
	if (use_ref) {
		device::UpdateNodeDistanceKernel<<<grid, blk, 0, stream>>>(
			m_reference_node_coords.DevicePtr(),
			m_node_status.Ptr(),
			m_node_distance.ptr(),
			m_node_size
		);
	}
	else {
		device::UpdateNodeDistanceKernel<<<grid, blk, 0, stream>>>(
			m_live_node_coords.DevicePtr(),
			m_node_status.Ptr(),
			m_node_distance.ptr(),
			m_node_size
		);
	}

	// Sync and check error
#if defined(CUDA_DEBUG_SYNC_CHECK)
	cudaSafeCall(cudaStreamSynchronize(stream));
	cudaSafeCall(cudaGetLastError());
#endif
}

void star::NodeGraph::ResetNodeGraphConnection(cudaStream_t stream) {
	dim3 blk(128);
	dim3 grid(divUp(m_node_size, blk.x));
	device::ResetNodeConnectionKernel<<<grid, blk, 0, stream>>>(
		m_node_knn_connect_weight.DevicePtr(),
		m_node_size
	);
}

void star::NodeGraph::ComputeNodeGraphConnectionFromSemantic(const floatX<d_max_num_semantic>& dynamic_regulation, cudaStream_t stream) {
	dim3 blk(128);
	dim3 grid(divUp(m_node_size, blk.x));
	device::ComputeNodeGraphConnectionFromSemanticKernel<<<grid, blk, 0, stream>>>(
		m_node_knn_connect_weight.DevicePtr(),
		m_node_knn.DevicePtr(),
		m_node_semantic_prob.Ptr(),
		dynamic_regulation,
		m_node_size
	);
}
void star::NodeGraph::updateAppendNodeInitialTime(
    const unsigned current_time, 
    const unsigned prev_node_size, 
    const unsigned append_node_size, 
    cudaStream_t stream
) {
	dim3 blk(128);
	dim3 grid(divUp(append_node_size, blk.x));
	device::UpdateAppendNodeInitialTimeKernel<<<grid, blk, 0, stream>>>(
		m_node_initial_time.Ptr(),
		current_time,
		prev_node_size,
		append_node_size
	);

	// Sync and check error
#if defined(CUDA_DEBUG_SYNC_CHECK)
	cudaSafeCall(cudaStreamSynchronize(stream));
	cudaSafeCall(cudaGetLastError());
#endif
	m_node_initial_time.ResizeArrayOrException(prev_node_size + append_node_size);
	return;
}

void star::NodeGraph::BuildNodeGraphFromDistance(
	cudaStream_t stream
) {
	// 1 - Update Knn based on distance
	dim3 blk(16, 16);
	dim3 grid(divUp(m_node_size, blk.x), divUp(m_node_size, blk.y));
	device::BuildNodeGraphKernel<<<grid, blk, 0, stream>>>(
		m_node_distance.ptr(),
		m_node_knn.DevicePtr(),
		m_node_knn_spatial_weight.DevicePtr(),
		m_node_size,
		m_node_radius * m_node_radius
	);
	
	// 2 - Compute NodeGraphPair & NodeList
	buildNodeGraphPair(stream);
	return;
}

void star::NodeGraph::BuildNodeGraphFromScratch(
	const bool use_ref,
	cudaStream_t stream
) {
	// Re-compute the distance
	UpdateNodeDistance(use_ref, stream);
	// Re-build patch
	BuildNodeGraphFromDistance(stream);
}