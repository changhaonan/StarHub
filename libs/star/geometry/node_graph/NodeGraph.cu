#include <star/math/vector_ops.hpp>
#include <star/geometry/constants.h>
#include <star/geometry/node_graph/NodeGraph.h>
#include <star/geometry/node_graph/brute_force_knn.cuh>
#include <device_launch_parameters.h>

namespace star::device
{

	__global__ void UpdateNodeDistanceKernel(
		const float4 *__restrict__ reference_node_coordinate,
		const uint2 *__restrict__ node_status,
		half *__restrict__ node_distance,
		const int node_size)
	{
		const auto idx = threadIdx.x + blockIdx.x * blockDim.x;
		const auto idy = threadIdx.y + blockIdx.y * blockDim.y;
		if (idx >= node_size || idy >= node_size || idx < idy)
			return;
		// Check status first
		half max_distance_history = 1e6f;

		// Compute distance
		if (node_status[idx].y != DELETED_NODE_STATUS && node_status[idy].y != DELETED_NODE_STATUS)
		{
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
		const half *__restrict__ distance,
		ushortX<d_node_knn_size> *__restrict__ node_knn,
		floatX<d_node_knn_size> *__restrict__ node_knn_spatial_weight,
		const unsigned node_size,
		const float node_radius_square)
	{
		const auto idx = threadIdx.x + blockIdx.x * blockDim.x;
		if (idx >= node_size)
			return;

		float nn_distance[d_node_knn_size];
		unsigned short nn_id[d_node_knn_size];
#pragma unroll
		for (unsigned i = 0; i < d_node_knn_size; ++i)
		{
			nn_distance[i] = 1e3f;
			nn_id[i] = 0;
		}

		KnnHeapExpandDevice<d_node_knn_size> knn_heap(nn_distance, nn_id);
		for (auto i = 0; i < node_size; ++i)
		{
			knn_heap.update(i, __half2float(distance[idx * d_max_num_nodes + i]));
		}
		knn_heap.sort();

		// Update nn idx
#pragma unroll
		for (unsigned i = 0; i < d_node_knn_size; ++i)
		{
			node_knn[idx][i] = knn_heap.index[i];
		}
		// Update spatial weight
#pragma unroll
		for (unsigned i = 0; i < d_node_knn_size; ++i)
		{
			node_knn_spatial_weight[idx][i] = __expf(-knn_heap.distance[i] / (2 * node_radius_square));
		}
#if defined(USE_INTERPOLATE_WEIGHT_NORMALIZATION)
		float weight_sum = 0.f;
#pragma unroll
		for (unsigned i = 0; i < d_node_knn_size; ++i)
		{
			weight_sum += node_knn_spatial_weight[idx][i];
		}
		const float inv_weight_sum = 1.0f / weight_sum;
#pragma unroll
		for (unsigned i = 0; i < d_node_knn_size; ++i)
		{
			node_knn_spatial_weight[idx][i] *= inv_weight_sum;
		}
#endif
	}

	__global__ void BuildNodeGraphPairKernel(
		const ushortX<d_node_knn_size> *__restrict__ node_knn,
		const uint2 *__restrict__ node_status,
		ushort3 *__restrict__ node_graph_pair,
		const unsigned node_size)
	{
		const auto idx = threadIdx.x + blockIdx.x * blockDim.x;
		if (idx >= node_size)
			return;
		const auto offset = d_node_knn_size * idx;
#pragma unroll
		for (auto i = 0; i < d_node_knn_size; ++i)
		{
			node_graph_pair[offset + i] = make_ushort3(idx, node_knn[idx][i], i);
		}
	}

	__global__ void UpdateAppendNodeInitialTimeKernel(
		unsigned *__restrict__ node_inital_time,
		const unsigned current_time,
		const unsigned prev_node_size,
		const unsigned append_node_size)
	{
		const auto idx = threadIdx.x + blockIdx.x * blockDim.x;
		if (idx >= append_node_size)
			return;
		node_inital_time[prev_node_size + idx] = current_time;
	}

	__global__ void UpdateAppendNodeDeformAccKernel(
		DualQuaternion *__restrict__ node_deform_acc,
		const unsigned prev_node_size,
		const unsigned append_node_size)
	{
		const auto idx = threadIdx.x + blockIdx.x * blockDim.x;
		if (idx >= append_node_size)
			return;
		node_deform_acc[prev_node_size + idx].set_identity();
	}

	__global__ void UpdateNodeDeformKernel(
		const DualQuaternion *__restrict__ delta_node_deform,
		DualQuaternion *__restrict__ node_deform,
		const unsigned num_nodes)
	{
		const auto idx = threadIdx.x + blockDim.x * blockIdx.x;
		if (idx >= num_nodes)
			return;
		auto prev_node_trans = mat34(node_deform[idx]).trans;
		node_deform[idx] = delta_node_deform[idx] * node_deform[idx];
	}

	__global__ void ResetNodeConnectionKernel(
		floatX<d_node_knn_size> *__restrict__ node_knn_connect_weight,
		const unsigned node_size)
	{
		const auto idx = threadIdx.x + blockIdx.x * blockDim.x;
		if (idx >= node_size)
			return;

#pragma unroll
		for (auto i = 0; i < d_node_knn_size; ++i)
		{
			node_knn_connect_weight[idx][i] = 1.f; // Reset as full connection
		}
	}

	__global__ void ComputeNodeGraphConnectionFromSemanticKernel(
		floatX<d_node_knn_size> *__restrict__ node_knn_connect_weight,
		const ushortX<d_node_knn_size> *__restrict__ node_knn,
		const ucharX<d_max_num_semantic> *__restrict__ node_semantic_prob,
		const floatX<d_max_num_semantic> dynamic_regulation,
		const unsigned node_size)
	{
		const auto idx = threadIdx.x + blockIdx.x * blockDim.x;
		if (idx >= node_size)
			return;

		// 1. Compare the semantic with neighbor
		int node_label = max_id(node_semantic_prob[idx]);
		for (auto i = 0; i < d_node_knn_size; ++i)
		{
			auto nn_idx = node_knn[idx][i];
			int nn_node_label = max_id(node_semantic_prob[nn_idx]);
			if (node_label != nn_node_label)
			{
				node_knn_connect_weight[idx][i] = 1e-12f; // Make it supper small
			}
			else
			{
				// Apply different regulation to different type of objects
				node_knn_connect_weight[idx][i] = dynamic_regulation[node_label];
			}
		}
	}

}

void star::NodeGraph::InitializeNodeGraphFromNode(
	const GArrayView<float4> &node_coords,
	const unsigned current_time,
	const bool use_ref,
	cudaStream_t stream)
{
	m_node_size = 0; // Set size to 0 first, because we are doing intialization
	updateNodeCoordinate(node_coords, current_time, stream);
	BuildNodeGraphFromScratch(use_ref, stream); // Use ref & live are the same here
}

void star::NodeGraph::InitializeNodeGraphFromNode(
	const std::vector<float4> &node_coords,
	const unsigned current_time,
	const bool use_ref,
	cudaStream_t stream)
{
	m_node_size = 0; // Set size to 0 first, because we are doing intialization
	updateNodeCoordinate(node_coords, current_time, stream);
	BuildNodeGraphFromScratch(use_ref, stream); // Use ref & live are the same here
}

void star::NodeGraph::updateNodeCoordinate(
	const GArrayView<float4> &node_coords, const unsigned current_time, cudaStream_t stream)
{
	cudaSafeCall(
		cudaMemcpyAsync(
			m_reference_node_coords.DevicePtr(),
			node_coords.Ptr(),
			node_coords.ByteSize(),
			cudaMemcpyDeviceToDevice,
			stream));
	// If the new nodes is more than previous nodes
	assert(node_coords.Size() > m_node_size); // Node graph size can only increase
	if (node_coords.Size() != m_node_size)
	{
		updateAppendNodeInitialTime(current_time, m_node_size, node_coords.Size() - m_node_size, stream);
		updateAppendNodeDeformAcc(m_node_size, node_coords.Size() - m_node_size, stream);
	}
	// Sync on device
	cudaSafeCall(cudaStreamSynchronize(stream));
	resizeNodeSize(node_coords.Size());
}

void star::NodeGraph::updateNodeCoordinate(
	const std::vector<float4> &node_coords, const unsigned current_time, cudaStream_t stream)
{
	// If the new nodes is more than previous nodes
	assert(node_coords.size() > m_node_size); // Node graph size can only increase
	if (node_coords.size() != m_node_size)
	{
		updateAppendNodeInitialTime(current_time, m_node_size, node_coords.size() - m_node_size, stream);
		updateAppendNodeDeformAcc(m_node_size, node_coords.size() - m_node_size, stream);
	}
	// Check the size
	STAR_CHECK(node_coords.size() <= d_max_num_nodes);
	// Copy the value to device
	cudaSafeCall(cudaMemcpyAsync(
		m_reference_node_coords.DevicePtr(),
		node_coords.data(),
		node_coords.size() * sizeof(float4),
		cudaMemcpyHostToDevice,
		stream));
	// Update size; Resize is only called here.
	cudaSafeCall(cudaStreamSynchronize(stream));
	resizeNodeSize(node_coords.size());
}

void star::NodeGraph::buildNodeGraphPair(cudaStream_t stream)
{
	// Clear first
	cudaSafeCall(cudaMemsetAsync((void *)m_node_graph_pair.Ptr(), 0, sizeof(ushort3) * d_node_knn_size * m_node_size, stream));

	dim3 blk(128);
	dim3 grid(divUp(m_node_size, blk.x));
	device::BuildNodeGraphPairKernel<<<grid, blk, 0, stream>>>(
		m_node_knn.DevicePtr(),
		m_node_status.Ptr(),
		m_node_graph_pair.Ptr(),
		m_node_size);

	// Sync and check error
#if defined(CUDA_DEBUG_SYNC_CHECK)
	cudaSafeCall(cudaStreamSynchronize(stream));
	cudaSafeCall(cudaGetLastError());
#endif

#ifdef DEBUG_NODE_GRAPH
	CheckNodeGraphPairConsistency(stream);
#endif
}

void star::NodeGraph::UpdateNodeDistance(const bool use_ref, cudaStream_t stream)
{
	dim3 blk(16, 16);
	dim3 grid(divUp(m_node_size, blk.x), divUp(m_node_size, blk.y));
	if (use_ref)
	{
		device::UpdateNodeDistanceKernel<<<grid, blk, 0, stream>>>(
			m_reference_node_coords.DevicePtr(),
			m_node_status.Ptr(),
			m_node_distance.ptr(),
			m_node_size);
	}
	else
	{
		device::UpdateNodeDistanceKernel<<<grid, blk, 0, stream>>>(
			m_live_node_coords.DevicePtr(),
			m_node_status.Ptr(),
			m_node_distance.ptr(),
			m_node_size);
	}

	// Sync and check error
#if defined(CUDA_DEBUG_SYNC_CHECK)
	cudaSafeCall(cudaStreamSynchronize(stream));
	cudaSafeCall(cudaGetLastError());
#endif
}

void star::NodeGraph::ResetNodeGraphConnection(cudaStream_t stream)
{
	dim3 blk(128);
	dim3 grid(divUp(m_node_size, blk.x));
	device::ResetNodeConnectionKernel<<<grid, blk, 0, stream>>>(
		m_node_knn_connect_weight.DevicePtr(),
		m_node_size);
}

void star::NodeGraph::ComputeNodeGraphConnectionFromSemantic(const floatX<d_max_num_semantic> &dynamic_regulation, cudaStream_t stream)
{
	dim3 blk(128);
	dim3 grid(divUp(m_node_size, blk.x));
	device::ComputeNodeGraphConnectionFromSemanticKernel<<<grid, blk, 0, stream>>>(
		m_node_knn_connect_weight.DevicePtr(),
		m_node_knn.DevicePtr(),
		m_node_semantic_prob.Ptr(),
		dynamic_regulation,
		m_node_size);
}
void star::NodeGraph::updateAppendNodeInitialTime(
	const unsigned current_time,
	const unsigned prev_node_size,
	const unsigned append_node_size,
	cudaStream_t stream)
{
	dim3 blk(128);
	dim3 grid(divUp(append_node_size, blk.x));
	device::UpdateAppendNodeInitialTimeKernel<<<grid, blk, 0, stream>>>(
		m_node_initial_time.Ptr(),
		current_time,
		prev_node_size,
		append_node_size);

	// Sync and check error
#if defined(CUDA_DEBUG_SYNC_CHECK)
	cudaSafeCall(cudaStreamSynchronize(stream));
	cudaSafeCall(cudaGetLastError());
#endif
	m_node_initial_time.ResizeArrayOrException(prev_node_size + append_node_size);
	return;
}

void star::NodeGraph::updateAppendNodeDeformAcc(
	const unsigned prev_node_size, const unsigned append_node_size, cudaStream_t stream)
{
	dim3 blk(128);
	dim3 grid(divUp(append_node_size, blk.x));
	device::UpdateAppendNodeDeformAccKernel<<<grid, blk, 0, stream>>>(
		m_node_deform_acc.Ptr(),
		prev_node_size,
		append_node_size);
	m_node_deform_acc.ResizeArrayOrException(prev_node_size + append_node_size);
}

void star::NodeGraph::UpdateNodeDeformAcc(
	const GArrayView<DualQuaternion> &delta_node_se3,
	cudaStream_t stream)
{
	STAR_CHECK_EQ(delta_node_se3.Size(), m_node_size);
	// Update node deform
	dim3 blk(128);
	dim3 grid(divUp(m_node_size, blk.x));
	device::UpdateNodeDeformKernel<<<grid, blk, 0, stream>>>(
		delta_node_se3.Ptr(),
		m_node_deform_acc.Ptr(),
		m_node_size);
}

void star::NodeGraph::BuildNodeGraphFromDistance(
	cudaStream_t stream)
{
	// 1 - Update Knn based on distance
	dim3 blk(16, 16);
	dim3 grid(divUp(m_node_size, blk.x), divUp(m_node_size, blk.y));
	device::BuildNodeGraphKernel<<<grid, blk, 0, stream>>>(
		m_node_distance.ptr(),
		m_node_knn.DevicePtr(),
		m_node_knn_spatial_weight.DevicePtr(),
		m_node_size,
		m_node_radius * m_node_radius);

	// 2 - Compute NodeGraphPair & NodeList
	buildNodeGraphPair(stream);
	return;
}

void star::NodeGraph::BuildNodeGraphFromScratch(
	const bool use_ref,
	cudaStream_t stream)
{
	// Re-compute the distance
	UpdateNodeDistance(use_ref, stream);
	// Re-build patch
	BuildNodeGraphFromDistance(stream);
}

star::NodeGraph::NodeGraph(const float node_radius) : m_node_size(0), m_node_radius(node_radius)
{
	// The other part of the constant memory should be filled with invalid points
	std::vector<float4> h_invalid_nodes;
	h_invalid_nodes.resize(d_max_num_nodes);
	float *begin = (float *)h_invalid_nodes.data();
	float *end = begin + 4 * size_t(d_max_num_nodes);
	std::fill(begin, end, 1e6f);

	// Allocate buffer
	m_reference_node_coords.AllocateBuffer(d_max_num_nodes);
	m_live_node_coords.AllocateBuffer(d_max_num_nodes);
	cudaSafeCall(cudaMalloc((void **)&m_newly_remove_count, sizeof(unsigned)));
	// Semantic (Optional)
	m_node_semantic_prob.AllocateBuffer(d_max_num_nodes);
	m_node_semantic_prob_vote_buffer.AllocateBuffer(size_t(d_max_num_nodes) * d_max_num_semantic);

	// To-remove
	m_node_status.AllocateBuffer(d_max_num_nodes);

	// KNN-related
	m_node_knn.AllocateBuffer(d_max_num_nodes);
	m_node_knn_spatial_weight.AllocateBuffer(d_max_num_nodes);
	m_node_knn_connect_weight.AllocateBuffer(d_max_num_nodes);

	const auto valid_pair_num = d_max_num_nodes * d_max_num_nodes;
	m_node_distance.create(valid_pair_num);

	// Optimization-related
	m_node_graph_pair.AllocateBuffer(size_t(d_max_num_nodes) * d_node_knn_size);

	// Voxel sampler
	m_vertex_subsampler = std::make_shared<VoxelSubsamplerSorting>();
	m_vertex_subsampler->AllocateBuffer(d_max_num_surfels);
	m_node_vertex_candidate.AllocateBuffer(d_max_num_surfel_candidates);

	// Node removal
	m_counter_node_outtrack.AllocateBuffer(d_max_num_nodes);
	m_node_initial_time.AllocateBuffer(d_max_num_nodes);

	// Auxilary
	m_node_deform_acc.AllocateBuffer(d_max_num_nodes);
}

star::NodeGraph::~NodeGraph()
{
	m_reference_node_coords.ReleaseBuffer();
	m_live_node_coords.ReleaseBuffer();
	m_node_status.ReleaseBuffer();

	m_node_distance.release();
	cudaSafeCall(cudaFree(m_newly_remove_count));

	m_vertex_subsampler->ReleaseBuffer();
	m_node_vertex_candidate.ReleaseBuffer();
	// Semantic (Optional)
	m_node_semantic_prob.ReleaseBuffer();
	m_node_semantic_prob_vote_buffer.ReleaseBuffer();

	// Node removal
	m_counter_node_outtrack.ReleaseBuffer();
	m_node_initial_time.ReleaseBuffer();

	// KNN-related
	m_node_knn.ReleaseBuffer();
	m_node_knn_spatial_weight.ReleaseBuffer();
	m_node_knn_connect_weight.ReleaseBuffer();

	// Opt-related
	m_node_graph_pair.ReleaseBuffer();

	// Auxilary
	m_node_deform_acc.ReleaseBuffer();
}

void star::NodeGraph::resizeNodeSize(unsigned node_size)
{
	// 1. Log the prev node size
	m_prev_node_size = m_node_size;

	// 2. Update new size && resize members
	m_node_size = node_size;
	m_reference_node_coords.ResizeArrayOrException(node_size);
	m_live_node_coords.ResizeArrayOrException(node_size);
	// Semantic (Optional)
	m_node_semantic_prob.ResizeArrayOrException(node_size);
	m_node_semantic_prob_vote_buffer.ResizeArrayOrException(size_t(node_size) * d_max_num_semantic);

	// KNN-related
	m_node_knn.ResizeArrayOrException(node_size);
	m_node_knn_spatial_weight.ResizeArrayOrException(node_size);
	m_node_knn_connect_weight.ResizeArrayOrException(node_size);

	m_node_graph_pair.ResizeArrayOrException(size_t(node_size) * d_node_knn_size);
	m_counter_node_outtrack.ResizeArrayOrException(node_size);
	m_node_initial_time.ResizeArrayOrException(node_size);
	m_node_status.ResizeArrayOrException(node_size);

	// Auxilary
	m_node_deform_acc.ResizeArrayOrException(node_size);
}

void star::NodeGraph::InitializeNodeGraphFromVertex(
	const GArrayView<float4> &vertex,
	const unsigned current_time,
	const bool use_ref,
	cudaStream_t stream)
{
	// 1. Clean previous node
	auto &reference_node = m_reference_node_coords.HostArray();
	reference_node.clear();
	auto &live_node = m_live_node_coords.HostArray();
	live_node.clear();

	// 2. Extract & Append
	AppendNodeFromVertexHost(
		vertex,
		reference_node,
		live_node,
		stream);

	// 3. Append 0 & 1 as anchor from the begining
	for (auto i = 0; i < 2; ++i)
	{
		const auto pos_anchor = make_float4(0.f, 0.f, 0.f, 1.f);
		reference_node.insert(reference_node.begin(), pos_anchor);
		live_node.insert(live_node.begin(), pos_anchor);
	}

	// 4. Synchronize to device
	m_reference_node_coords.SyncToDevice(stream);
	m_live_node_coords.SyncToDevice(stream);
	resizeNodeSize(m_reference_node_coords.DeviceArraySize());

	// 5. Compute graph
	updateAppendNodeInitialTime(current_time, 0, m_node_size, stream);
	updateAppendNodeDeformAcc(0, m_node_size, stream);
	BuildNodeGraphFromScratch(use_ref, stream); // Use live & ref are the same here, use live

	// 6. Sync & resize
	cudaSafeCall(cudaStreamSynchronize(stream));
}

void star::NodeGraph::NaiveExpandNodeGraphFromVertexUnsupported(
	const GArrayView<float4> &vertex_unsupported,
	const unsigned current_time,
	cudaStream_t stream)
{
	// 0. Pre-check
	if (vertex_unsupported.Size() == 0)
	{
		printf("No unsupported vertex from measurement, so NodeGraph stays the same.\n");
		return;
	}
	// 1. Get host node
	m_reference_node_coords.SyncToHost(stream, true);
	m_live_node_coords.SyncToHost(stream, true);
	auto &reference_node = m_reference_node_coords.HostArray();
	auto &live_node = m_live_node_coords.HostArray();
	unsigned prev_node_size = reference_node.size();

	// 2. Extract & Append
	AppendNodeFromVertexHost(
		vertex_unsupported,
		reference_node,
		live_node,
		stream);

	// 4. Synchronize to device
	m_reference_node_coords.SyncToDevice(stream);
	m_live_node_coords.SyncToDevice(stream);
	resizeNodeSize(m_reference_node_coords.DeviceArraySize());

	// 5. Update Auxilary
	updateAppendNodeInitialTime(current_time, prev_node_size, m_node_size - prev_node_size, stream);
	updateAppendNodeDeformAcc(prev_node_size, m_node_size - prev_node_size, stream);
}

star::NodeGraph4Skinner star::NodeGraph::GenerateNodeGraph4Skinner() const
{
	NodeGraph4Skinner node_graph4skinner;
	node_graph4skinner.reference_node_coords = m_reference_node_coords.DeviceArrayReadOnly();
	node_graph4skinner.live_node_coords = m_live_node_coords.DeviceArrayReadOnly();
	node_graph4skinner.node_status = m_node_status.View();
	node_graph4skinner.node_knn = m_node_knn.DeviceArrayReadOnly();
	node_graph4skinner.node_knn_spatial_weight = m_node_knn_spatial_weight.DeviceArrayReadOnly();
	node_graph4skinner.node_knn_connect_weight = m_node_knn_connect_weight.DeviceArrayReadOnly();
	node_graph4skinner.node_semantic_prob = m_node_semantic_prob.View();
	node_graph4skinner.node_radius_square = m_node_radius * m_node_radius;
	return node_graph4skinner;
}

star::NodeGraph::NodeGraph4Solver star::NodeGraph::GenerateNodeGraph4Solver() const
{
	NodeGraph4Solver node_graph4solver;
	node_graph4solver.reference_node_coords = m_reference_node_coords.DeviceArrayReadOnly();
	node_graph4solver.node_graph = m_node_graph_pair.View();
	node_graph4solver.nodel_knn = m_node_knn.DeviceArrayReadOnly();
	node_graph4solver.node_knn_connect_weight = m_node_knn_connect_weight.DeviceArrayReadOnly();
	node_graph4solver.node_knn_spatial_weight = m_node_knn_spatial_weight.DeviceArrayReadOnly();
	node_graph4solver.num_node = m_node_size;
	node_graph4solver.node_radius_square = m_node_radius * m_node_radius;
	return node_graph4solver;
}

star::WarpField::DeformationAcess star::NodeGraph::DeformAccess()
{
	WarpField::DeformationAcess deform_access;
	deform_access.node_knn = m_node_knn.DeviceArrayReadOnly();
	deform_access.node_knn_spatial_weight = m_node_knn_spatial_weight.DeviceArrayReadOnly();
	deform_access.node_knn_connect_weight = m_node_knn_connect_weight.DeviceArrayReadOnly();

	deform_access.reference_node_coords = m_reference_node_coords.DeviceArrayReadWrite();
	deform_access.live_node_coords = m_live_node_coords.DeviceArrayReadWrite();

	return deform_access;
}

void star::NodeGraph::ReAnchor(
	NodeGraph::ConstPtr src_node_graph,
	NodeGraph::Ptr tar_node_graph,
	cudaStream_t stream)
{
	// Copy KNN structure
	cudaSafeCall(
		cudaMemcpyAsync(
			tar_node_graph->m_node_knn.DevicePtr(),
			src_node_graph->m_node_knn.DevicePtr(),
			src_node_graph->m_node_knn.DeviceArrayReadOnly().ByteSize(),
			cudaMemcpyDeviceToDevice, stream));
	cudaSafeCall(
		cudaMemcpyAsync(
			tar_node_graph->m_node_knn_spatial_weight.DevicePtr(),
			src_node_graph->m_node_knn_spatial_weight.DevicePtr(),
			src_node_graph->m_node_knn_spatial_weight.DeviceArrayReadOnly().ByteSize(),
			cudaMemcpyDeviceToDevice, stream));
	cudaSafeCall(
		cudaMemcpyAsync(
			tar_node_graph->m_node_knn_connect_weight.DevicePtr(),
			src_node_graph->m_node_knn_connect_weight.DevicePtr(),
			src_node_graph->m_node_knn_connect_weight.DeviceArrayReadOnly().ByteSize(),
			cudaMemcpyDeviceToDevice, stream));
	cudaSafeCall(
		cudaMemcpyAsync(
			tar_node_graph->m_node_graph_pair.Ptr(),
			src_node_graph->m_node_graph_pair.Ptr(),
			src_node_graph->m_node_graph_pair.View().ByteSize(),
			cudaMemcpyDeviceToDevice, stream));

	// Semantic Prob (Optional)
	cudaSafeCall(
		cudaMemcpyAsync(
			tar_node_graph->m_node_semantic_prob.Ptr(),
			src_node_graph->m_node_semantic_prob.Ptr(),
			src_node_graph->m_node_semantic_prob.ArrayByteSize(),
			cudaMemcpyDeviceToDevice, stream));

	// Auxilary
	cudaSafeCall(
		cudaMemcpyAsync(
			tar_node_graph->m_node_status.Ptr(),
			src_node_graph->m_node_status.Ptr(),
			src_node_graph->m_node_status.View().ByteSize(),
			cudaMemcpyDeviceToDevice, stream));
	cudaSafeCall(
		cudaMemcpyAsync(
			tar_node_graph->m_counter_node_outtrack.Ptr(),
			src_node_graph->m_counter_node_outtrack.Ptr(),
			src_node_graph->m_counter_node_outtrack.View().ByteSize(),
			cudaMemcpyDeviceToDevice, stream));
	cudaSafeCall(
		cudaMemcpyAsync(
			tar_node_graph->m_node_initial_time.Ptr(),
			src_node_graph->m_node_initial_time.Ptr(),
			src_node_graph->m_node_initial_time.View().ByteSize(),
			cudaMemcpyDeviceToDevice, stream));
	cudaSafeCall(
		cudaMemcpyAsync(
			tar_node_graph->m_node_distance.ptr(),
			src_node_graph->m_node_distance.ptr(),
			src_node_graph->m_node_distance.size() * sizeof(half),
			cudaMemcpyDeviceToDevice, stream));
	cudaSafeCall(
		cudaMemcpyAsync(
			tar_node_graph->m_node_deform_acc.Ptr(),
			src_node_graph->m_node_deform_acc.Ptr(),
			src_node_graph->m_node_deform_acc.ArrayByteSize(),
			cudaMemcpyDeviceToDevice, stream));

	// Geometry
	cudaSafeCall(
		cudaMemcpyAsync(
			tar_node_graph->m_reference_node_coords.DevicePtr(),
			src_node_graph->m_live_node_coords.DevicePtr(),
			src_node_graph->m_live_node_coords.DeviceArrayReadOnly().ByteSize(),
			cudaMemcpyDeviceToDevice, stream));

	// Sync & Resize
	cudaSafeCall(cudaStreamSynchronize(stream));
	tar_node_graph->resizeNodeSize(src_node_graph->GetNodeSize());
	tar_node_graph->m_prev_node_size = src_node_graph->m_prev_node_size; // Copy history
}

void star::NodeGraph::AppendNodeFromVertexHost(
	const GArrayView<float4> &vertex,
	std::vector<float4> &reference_node,
	std::vector<float4> &live_node,
	cudaStream_t stream)
{
	if (vertex.Size() == 0)
	{
		printf("No unsupported surfel from measurement.\n");
		return;
	}
	// Select candidate first
	const auto subsample_voxel = 0.7f * m_node_radius;
	m_vertex_subsampler->PerformSubsample(
		vertex,
		m_node_vertex_candidate,
		subsample_voxel, stream);

	// Overlap checking selection
	const float sample_distance_square = (0.85f * m_node_radius) * (0.85f * m_node_radius);
	auto &node_vertex_candidate = m_node_vertex_candidate.HostArray();

	std::vector<float4> appended_node;
	for (auto i = 0; i < node_vertex_candidate.size(); i++)
	{
		const float4 point = make_float4(node_vertex_candidate[i].x, node_vertex_candidate[i].y, node_vertex_candidate[i].z, 1.0f);
		if (std::isnan(point.x) || std::isnan(point.y) || std::isnan(point.z))
		{
			LOG(FATAL) << "Nan in node candidate";
			continue;
		}

		// Brute-force check
		bool is_node = true;
		for (auto j = 0; j < appended_node.size(); j++)
		{
			const auto &node_vertex = appended_node[j];
			if (squared_norm(node_vertex - point) <= sample_distance_square)
			{
				is_node = false;
				break;
			}
		}

		// If this is node
		if (is_node)
		{
			reference_node.push_back(point);
			live_node.push_back(point);
			appended_node.push_back(point);
		}
	}
}