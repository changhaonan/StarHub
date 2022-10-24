#include "star/geometry/node_graph/NodeGraph.h"

#ifdef DEBUG_NODE_GRAPH
#include "context.hpp"
#endif

star::NodeGraph::NodeGraph(const float node_radius) : m_node_size(0), m_node_radius(node_radius) {
	// The other part of the constant memory should be filled with invalid points
	std::vector<float4> h_invalid_nodes;
	h_invalid_nodes.resize(Constants::kMaxNumNodes);
	float* begin = (float*)h_invalid_nodes.data();
	float* end = begin + 4 * size_t(Constants::kMaxNumNodes);
	std::fill(begin, end, 1e6f);

	// Allocate buffer
	m_reference_node_coords.AllocateBuffer(Constants::kMaxNumNodes);
	m_live_node_coords.AllocateBuffer(Constants::kMaxNumNodes);
	cudaSafeCall(cudaMalloc((void**)&m_newly_remove_count, sizeof(unsigned)));
	// Semantic (Optional)
	m_node_semantic_prob.AllocateBuffer(Constants::kMaxNumNodes);
	m_node_semantic_prob_vote_buffer.AllocateBuffer(size_t(Constants::kMaxNumNodes) * d_max_num_semantic);

	// To-remove
	m_node_status.AllocateBuffer(Constants::kMaxNumNodes);

	// KNN-related
	m_node_knn.AllocateBuffer(Constants::kMaxNumNodes);
	m_node_knn_spatial_weight.AllocateBuffer(Constants::kMaxNumNodes);
	m_node_knn_connect_weight.AllocateBuffer(Constants::kMaxNumNodes);

	const auto valid_pair_num = Constants::kMaxNumNodes * Constants::kMaxNumNodes;
	m_node_distance.create(valid_pair_num);
	
	// Optimization-related
	m_node_graph_pair.AllocateBuffer(size_t(Constants::kMaxNumNodes) * d_node_knn_size);

	// Voxel sampler
	m_vertex_subsampler = std::make_shared<VoxelSubsamplerSorting>();
	m_vertex_subsampler->AllocateBuffer(Constants::kMaxNumSurfels);
	m_node_vertex_candidate.AllocateBuffer(Constants::kMaxNumSurfelCandidates);

	// Node removal
	m_counter_node_outtrack.AllocateBuffer(Constants::kMaxNumNodes);
    m_node_initial_time.AllocateBuffer(Constants::kMaxNumNodes);
}

star::NodeGraph::~NodeGraph() {
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
}

void star::NodeGraph::resizeNodeSize(unsigned node_size) {
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
}

void star::NodeGraph::InitializeNodeGraphFromVertex(
	const GArrayView<float4>& vertex,
    const unsigned current_time,
	const bool use_ref,
	cudaStream_t stream) {
	// 1. Clean previous node
	auto& reference_node = m_reference_node_coords.HostArray(); reference_node.clear();
	auto& live_node = m_live_node_coords.HostArray(); live_node.clear();

	// 2. Extract & Append
	AppendNodeFromVertexHost(
		vertex,
		reference_node,
		live_node,
		stream
	);
    
    // 3. Append 0 & 1 as anchor from the begining
    for (auto i = 0; i < 2; ++i) {
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
	BuildNodeGraphFromScratch(use_ref, stream);  // Use live & ref are the same here, use live

	// 6. Sync & resize
	cudaSafeCall(cudaStreamSynchronize(stream));
}

void star::NodeGraph::NaiveExpandNodeGraphFromVertexUnsupported(
	const GArrayView<float4>& vertex_unsupported,
	const unsigned current_time,
	cudaStream_t stream
) {
	// 0. Pre-check
	if (vertex_unsupported.Size() == 0) {
		printf("No unsupported vertex from measurement.\n");
		return;
	}
	// 1. Get host node
	m_reference_node_coords.SyncToHost(stream, true);
	m_live_node_coords.SyncToHost(stream, true);
	auto& reference_node = m_reference_node_coords.HostArray();
	auto& live_node = m_live_node_coords.HostArray();
	unsigned prev_node_size = reference_node.size();

	// 2. Extract & Append
	AppendNodeFromVertexHost(
		vertex_unsupported,
		reference_node,
		live_node,
		stream
	);

	// 4. Synchronize to device
	m_reference_node_coords.SyncToDevice(stream);
	m_live_node_coords.SyncToDevice(stream);
	resizeNodeSize(m_reference_node_coords.DeviceArraySize());

	// 5. Update graph time
	updateAppendNodeInitialTime(current_time, prev_node_size, m_node_size - prev_node_size, stream);
}

star::NodeGraph4Skinner star::NodeGraph::GenerateNodeGraph4Skinner() const {
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

star::NodeGraph4Solver star::NodeGraph::GenerateNodeGraph4Solver() const {
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

star::WarpField::DeformationAcess star::NodeGraph::DeformAccess() {
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
	cudaStream_t stream
) {
	// Copy KNN structure
	cudaSafeCall(
		cudaMemcpyAsync(
			tar_node_graph->m_node_knn.DevicePtr(),
			src_node_graph->m_node_knn.DevicePtr(),
			src_node_graph->m_node_knn.DeviceArrayReadOnly().ByteSize(),
			cudaMemcpyDeviceToDevice, stream
		)
	);
	cudaSafeCall(
		cudaMemcpyAsync(
			tar_node_graph->m_node_knn_spatial_weight.DevicePtr(),
			src_node_graph->m_node_knn_spatial_weight.DevicePtr(),
			src_node_graph->m_node_knn_spatial_weight.DeviceArrayReadOnly().ByteSize(),
			cudaMemcpyDeviceToDevice, stream
		)
	);
	cudaSafeCall(
		cudaMemcpyAsync(
			tar_node_graph->m_node_knn_connect_weight.DevicePtr(),
			src_node_graph->m_node_knn_connect_weight.DevicePtr(),
			src_node_graph->m_node_knn_connect_weight.DeviceArrayReadOnly().ByteSize(),
			cudaMemcpyDeviceToDevice, stream
		)
	);
	cudaSafeCall(
		cudaMemcpyAsync(
			tar_node_graph->m_node_graph_pair.Ptr(),
			src_node_graph->m_node_graph_pair.Ptr(),
			src_node_graph->m_node_graph_pair.View().ByteSize(),
			cudaMemcpyDeviceToDevice, stream
		)
	);

	// Semantic Prob (Optional)
	cudaSafeCall(
		cudaMemcpyAsync(
			tar_node_graph->m_node_semantic_prob.Ptr(),
			src_node_graph->m_node_semantic_prob.Ptr(),
			src_node_graph->m_node_semantic_prob.ArrayByteSize(),
			cudaMemcpyDeviceToDevice, stream
		)
	);

	// Auxilary
	cudaSafeCall(
		cudaMemcpyAsync(
			tar_node_graph->m_node_status.Ptr(),
			src_node_graph->m_node_status.Ptr(),
			src_node_graph->m_node_status.View().ByteSize(),
			cudaMemcpyDeviceToDevice, stream
		)
	);
	cudaSafeCall(
		cudaMemcpyAsync(
			tar_node_graph->m_counter_node_outtrack.Ptr(),
			src_node_graph->m_counter_node_outtrack.Ptr(),
			src_node_graph->m_counter_node_outtrack.View().ByteSize(),
			cudaMemcpyDeviceToDevice, stream
		)
	);
	cudaSafeCall(
		cudaMemcpyAsync(
			tar_node_graph->m_node_initial_time.Ptr(),
			src_node_graph->m_node_initial_time.Ptr(),
			src_node_graph->m_node_initial_time.View().ByteSize(),
			cudaMemcpyDeviceToDevice, stream
		)
	);
	cudaSafeCall(
		cudaMemcpyAsync(
			tar_node_graph->m_node_distance.ptr(),
			src_node_graph->m_node_distance.ptr(),
			src_node_graph->m_node_distance.size() * sizeof(half),
			cudaMemcpyDeviceToDevice, stream
		)
	);
	// Geometry
	cudaSafeCall(
		cudaMemcpyAsync(
			tar_node_graph->m_reference_node_coords.DevicePtr(),
			src_node_graph->m_live_node_coords.DevicePtr(),
			src_node_graph->m_live_node_coords.DeviceArrayReadOnly().ByteSize(),
			cudaMemcpyDeviceToDevice, stream
		)
	);

	// Sync & Resize
	cudaSafeCall(cudaStreamSynchronize(stream));
	tar_node_graph->resizeNodeSize(src_node_graph->GetNodeSize());
	tar_node_graph->m_prev_node_size = src_node_graph->m_prev_node_size;  // Copy history
}

void star::NodeGraph::AppendNodeFromVertexHost(
	const GArrayView<float4>& vertex,
	std::vector<float4>& reference_node,
	std::vector<float4>& live_node,
	cudaStream_t stream
) {
	if (vertex.Size() == 0) {
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
	auto& node_vertex_candidate = m_node_vertex_candidate.HostArray();

	std::vector<float4> appended_node;
	for (auto i = 0; i < node_vertex_candidate.size(); i++) {
		const float4 point = make_float4(node_vertex_candidate[i].x, node_vertex_candidate[i].y, node_vertex_candidate[i].z, 1.0f);
		if (std::isnan(point.x) || std::isnan(point.y) || std::isnan(point.z)) {
			LOG(FATAL) << "Nan in node candidate";
			continue;
		}

		// Brute-force check
		bool is_node = true;
		for (auto j = 0; j < appended_node.size(); j++) {
			const auto& node_vertex = appended_node[j];
			if (squared_norm(node_vertex - point) <= sample_distance_square) {
				is_node = false;
				break;
			}
		}

		// If this is node
		if (is_node) {
			reference_node.push_back(point);
			live_node.push_back(point);
			appended_node.push_back(point);
		}
	}
}