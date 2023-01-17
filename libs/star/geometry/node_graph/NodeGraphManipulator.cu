#include "star/geometry/node_graph/NodeGraphManipulator.h"
#include "star/geometry/node_graph/brute_force_knn.cuh"
#include <device_launch_parameters.h>

namespace star::device
{
	__global__ void ComputeSurfelSupporterKernel(
		const float4 *__restrict__ vertex_confid_candidate, // Candidate
		const float4 *__restrict__ node_coord,
		const uint2 *__restrict__ node_status,					// 0 is active, positive is frozen, negative is deleted
		unsigned *__restrict__ candidate_validity_indicator,	// If candidate is supported by an active node or no support
		unsigned *__restrict__ candidate_unsupported_indicator, // If candidate is supported
		ushortX<d_surfel_knn_size> *__restrict__ candidate_knn,
		const unsigned node_size,
		const unsigned candidate_size,
		const float node_radius_square)
	{
		const auto idx = threadIdx.x + blockIdx.x * blockDim.x;
		if (idx >= candidate_size)
			return;
		auto v = vertex_confid_candidate[idx];

		// 1. Compute the nearest node
		floatX<d_surfel_knn_size> distance;
		ushortX<d_surfel_knn_size> index;
#pragma unroll
		for (auto i = 0; i < d_surfel_knn_size; ++i)
		{
			distance[i] = 1e6f;
			index[i] = 0;
		}

		KnnHeapExpandDevice<d_surfel_knn_size> knn_heap(distance, index); // Head-only
		// 2. Traverse all existing nodes
		for (auto i = 0; i < node_size; ++i)
		{
			float dist = 1e6f;
			// 2.1. Check node_status & node_semantic
			if (node_status[i].y != DELETED_NODE_STATUS)
			{
				float4 v_node = node_coord[i];
				float dx = (v.x - v_node.x);
				float dy = (v.y - v_node.y);
				float dz = (v.z - v_node.z);
				dist = dx * dx + dy * dy + dz * dz;
			}
			knn_heap.update(i, dist);
		}
		knn_heap.sort();
		index = knn_heap.index;
		distance = knn_heap.distance;

		// 2. Check if the nearest is a frozen one & within node_radius_square
		candidate_unsupported_indicator[idx] = unsigned(distance[0] >= node_radius_square);
		candidate_validity_indicator[idx] = unsigned(!(distance[0] < node_radius_square && (node_status[index[0]].y != 0) && (node_status[index[0]].y != DELETED_NODE_STATUS)));
		candidate_knn[idx] = index;
	}

	__global__ void UpdateCounterNodeOutTrackKernel(
		const unsigned *__restrict__ surfel_validity,
		const ushortX<d_surfel_knn_size> *__restrict__ surfel_knn,
		unsigned *__restrict__ counter_node_outtrack,
		const unsigned surfel_size)
	{
		const auto idx = threadIdx.x + blockIdx.x * blockDim.x;
		if (idx >= surfel_size)
			return;
		if (!surfel_validity[idx])
		{
			auto nearest_id = surfel_knn[idx][0]; // Nearest id of the invalid surfel
			atomicAdd(&counter_node_outtrack[nearest_id], (unsigned)1);
		}
	}

	__global__ void RemoveNodeOutTrackKernel(
		const ushortX<d_node_knn_size> *__restrict__ node_knn,
		const unsigned *__restrict__ counter_node_outtrack,
		uint2 *__restrict__ node_status,
		half *__restrict__ node_distance,
		unsigned *newly_remove_count,
		const unsigned node_size,
		const float counter_node_outtrack_threshold,
		const unsigned frozen_time)
	{
		const auto idx = threadIdx.x + blockIdx.x * blockDim.x;
		if (idx >= node_size)
			return;

		if (counter_node_outtrack[idx] >= counter_node_outtrack_threshold &&
			node_status[idx].y != DELETED_NODE_STATUS)
		{ // Set new false, old don't change
			atomicOr(&node_status[idx].y, DELETED_NODE_STATUS);
			atomicAdd(newly_remove_count, (unsigned)1); // Remove one
			// Set all related distances to 1e6
			for (auto i = 0; i < node_size; ++i)
			{
				const auto offset_xy = idx * d_max_num_nodes + i;
				const auto offset_yx = i * d_max_num_nodes + idx;
				node_distance[offset_xy] = 1e6f;
				node_distance[offset_yx] = 1e6f;
			}

#define USE_NEIGHBOR_FROZEN_LAYER_ONE
#if defined(USE_NEIGHBOR_FROZEN_LAYER_ONE)
			// Set neighbor frozen time at least to be 10
			//  FIXME: Do we need to consider the co-visit?
			auto knn = node_knn[idx];
#pragma unroll
			for (auto i = 0; i < d_node_knn_size; ++i)
			{
				atomicOr(&node_status[knn[i]].y, frozen_time);
			}
#endif
		}
	}

	__global__ void NodeSemanticProbVoteKernel(
		const ucharX<d_max_num_semantic> *__restrict__ surfel_semantic_prob,
		float *__restrict__ node_patch_semantic_prob_vote,
		const ushortX<d_surfel_knn_size> *__restrict__ surfel_knn,
		const unsigned surfel_size)
	{
		const auto idx = threadIdx.x + blockDim.x * blockIdx.x;
		if (idx >= surfel_size)
			return;
		const auto nearest_node_idx = surfel_knn[idx][0];

		// Voting the semantic
		auto semantic_prob = surfel_semantic_prob[idx];
#pragma unroll
		for (auto i = 0; i < d_max_num_semantic; ++i)
		{
			float semantic_prob_inc = float(semantic_prob[i]) / 255.f;
			atomicAdd(&node_patch_semantic_prob_vote[nearest_node_idx * d_max_num_semantic + i], semantic_prob_inc);
		}
	}

	__global__ void NodeSemanticProbAverageKernel(
		const float *__restrict__ node_patch_semantic_prob_vote,
		ucharX<d_max_num_semantic> *__restrict__ node_semantic_prob,
		const unsigned node_size,
		const unsigned num_node_offset = 0)
	{
		const auto idx = threadIdx.x + blockDim.x * blockIdx.x;
		if (idx >= (node_size - num_node_offset))
			return;
		// 1. Compute the voting
		float semantic_prob_sum = 0.f;
		const float *node_patch_semantic_prob_vote_idx = &(node_patch_semantic_prob_vote[(idx + num_node_offset) * d_max_num_semantic]);
#pragma unroll
		for (auto i = 0; i < d_max_num_semantic; ++i)
		{
			semantic_prob_sum += node_patch_semantic_prob_vote_idx[i];
		}

		// 2. Update the node semantic
		ucharX<d_max_num_semantic> node_semantic_prob_idx;
#pragma unroll
		for (auto i = 0; i < d_max_num_semantic; ++i)
		{
			float node_semantic_prob_idx_i = node_patch_semantic_prob_vote_idx[i] / semantic_prob_sum * 255.f;
			// Replace with new one (Average?)
			node_semantic_prob_idx[i] = (unsigned char)node_semantic_prob_idx_i;
		}

		node_semantic_prob[(idx + num_node_offset)] = node_semantic_prob_idx;
	}

	__global__ void FilterNodeBySemanticAtomicKernel(
		const ucharX<d_max_num_semantic> *__restrict__ node_semantic_prob,
		const unsigned short semantic_selected,
		unsigned short *__restrict__ node_list_filtered,
		int *__restrict__ count,
		const unsigned num_nodes)
	{
		const auto idx = threadIdx.x + blockDim.x * blockIdx.x;
		if (idx >= num_nodes)
			return;
		ucharX<d_max_num_semantic> node_semantic_prob_idx = node_semantic_prob[idx];
		auto semantic_id = max_id(node_semantic_prob_idx);
		if (semantic_id == semantic_selected)
		{
			auto old_count = atomicAdd(count, 1);
			node_list_filtered[old_count] = idx;
		}
	}
}

void star::NodeGraphManipulator::CheckSurfelCandidateSupportStatus(
	const GArrayView<float4> &vertex_confid_candidate,
	const GArrayView<float4> &node_coord,
	const GArrayView<uint2> &node_status,
	GArraySlice<unsigned> candidate_validity_indicator,
	GArraySlice<unsigned> candidate_unsupported_indicator,
	GArraySlice<ushortX<d_surfel_knn_size>> candidate_knn,
	cudaStream_t stream,
	const float node_radius_square)
{
	unsigned num_candidate = vertex_confid_candidate.Size();
	unsigned num_node = node_coord.Size();
	dim3 blk(128);
	dim3 grid(divUp(num_candidate, blk.x));
	device::ComputeSurfelSupporterKernel<<<grid, blk, 0, stream>>>(
		vertex_confid_candidate.Ptr(), // Candidate
		node_coord.Ptr(),
		node_status.Ptr(),					   // 0 is active, positive is frozen, negative is deleted
		candidate_validity_indicator.Ptr(),	   // If candidate is supported by an active node or no support
		candidate_unsupported_indicator.Ptr(), // If candidate is supported
		candidate_knn.Ptr(),
		num_node,
		num_candidate,
		node_radius_square);
}

void star::NodeGraphManipulator::UpdateCounterNodeOutTrack(
	const GArrayView<unsigned> &surfel_validity,
	const GArrayView<ushortX<d_surfel_knn_size>> &surfel_knn,
	GArraySlice<unsigned> counter_node_outtrack,
	cudaStream_t stream)
{
	unsigned num_surfel = surfel_validity.Size();
	dim3 blk(128);
	dim3 grid(divUp(num_surfel, blk.x));
	device::UpdateCounterNodeOutTrackKernel<<<grid, blk, 0, stream>>>(
		surfel_validity.Ptr(),
		surfel_knn.Ptr(),
		counter_node_outtrack.Ptr(),
		num_surfel);
}

void star::NodeGraphManipulator::RemoveNodeOutTrackSync(
	const GArrayView<ushortX<d_surfel_knn_size>> &node_knn,
	const GArrayView<unsigned> &counter_node_outtrack,
	GArraySlice<uint2> node_status,
	GArraySlice<half> node_distance,
	unsigned &num_node_remove_count,
	const float counter_node_outtrack_threshold,
	const unsigned frozen_time,
	cudaStream_t stream)
{
	// Create local variable
	unsigned *newly_remove_count;
	cudaSafeCall(cudaMallocAsync((void **)&newly_remove_count, sizeof(unsigned), stream));
	cudaSafeCall(cudaMemsetAsync((void *)newly_remove_count, 0, sizeof(unsigned), stream));

	auto node_size = node_status.Size();
	dim3 blk(128);
	dim3 grid(divUp(node_size, blk.x));
	device::RemoveNodeOutTrackKernel<<<grid, blk, 0, stream>>>(
		node_knn.Ptr(),
		counter_node_outtrack.Ptr(),
		node_status.Ptr(),
		node_distance.Ptr(),
		newly_remove_count,
		node_size,
		counter_node_outtrack_threshold,
		frozen_time);
	cudaSafeCall(cudaMemcpyAsync(
		&num_node_remove_count,
		newly_remove_count,
		sizeof(unsigned),
		cudaMemcpyDeviceToHost,
		stream));
	cudaSafeCall(cudaStreamSynchronize(stream)); // Sync before exist
	cudaSafeCall(cudaFree(newly_remove_count));
}

void star::NodeGraphManipulator::UpdateNodeSemanticProb(
	const GArrayView<ushortX<d_surfel_knn_size>> &surfel_knn,
	const GArrayView<ucharX<d_max_num_semantic>> &surfel_semantic_prob,
	GArraySlice<ucharX<d_max_num_semantic>> node_semantic_prob,
	GArraySlice<float> node_semantic_prob_vote_buffer,
	cudaStream_t stream)
{
	// 1. Clean the voting buffer
	cudaSafeCall(cudaMemsetAsync(node_semantic_prob_vote_buffer.Ptr(), 0, node_semantic_prob_vote_buffer.ByteSize(), stream));
	const auto surfel_size = surfel_knn.Size();
	const auto node_size = node_semantic_prob.Size();

	// 2. Voting
	dim3 blk(128);
	dim3 grid(divUp(surfel_size, blk.x));
	device::NodeSemanticProbVoteKernel<<<grid, blk, 0, stream>>>(
		surfel_semantic_prob.Ptr(),
		node_semantic_prob_vote_buffer.Ptr(),
		surfel_knn.Ptr(),
		surfel_knn.Size());

	// 3. Averaging
	blk = dim3(128);
	grid = dim3(divUp(node_size, blk.x));
	device::NodeSemanticProbAverageKernel<<<grid, blk, 0, stream>>>(
		node_semantic_prob_vote_buffer.Ptr(),
		node_semantic_prob.Ptr(),
		node_size);
}

void star::NodeGraphManipulator::UpdateIncNodeSemanticProb(
	const GArrayView<ushortX<d_surfel_knn_size>> &surfel_knn,
	const GArrayView<ucharX<d_max_num_semantic>> &surfel_semantic_prob,
	GArraySlice<ucharX<d_max_num_semantic>> node_semantic_prob,
	GArraySlice<float> node_semantic_prob_vote_buffer,
	unsigned num_prev_node,
	cudaStream_t stream)
{
	// 1. Clean the voting buffer
	cudaSafeCall(cudaMemsetAsync(node_semantic_prob_vote_buffer.Ptr(), 0, node_semantic_prob_vote_buffer.ByteSize(), stream));
	const auto surfel_size = surfel_knn.Size();
	const auto node_size = node_semantic_prob.Size();

	// 2. Voting
	dim3 blk(128);
	dim3 grid(divUp(surfel_size, blk.x));
	device::NodeSemanticProbVoteKernel<<<grid, blk, 0, stream>>>(
		surfel_semantic_prob.Ptr(),
		node_semantic_prob_vote_buffer.Ptr(),
		surfel_knn.Ptr(),
		surfel_knn.Size());

	// 3. Averaging
	blk = dim3(128);
	grid = dim3(divUp(node_size, blk.x));
	device::NodeSemanticProbAverageKernel<<<grid, blk, 0, stream>>>(
		node_semantic_prob_vote_buffer.Ptr(),
		node_semantic_prob.Ptr(),
		node_size,
		num_prev_node);
}

void star::NodeGraphManipulator::AvergeNodeMovementAndPos(
	const GArrayView<float4> &node_coord,
	const GArrayView<DualQuaternion> &delta_node_deform,
	const std::vector<unsigned short> &node_list,
	GArraySlice<DualQuaternion> node_deform,
	DualQuaternion &average_node_se3,
	float3 &average_node_pos)
{
	// Average the selected nodes
	std::vector<float4> h_node_coord;
	node_coord.Download(h_node_coord);
	std::vector<DualQuaternion> h_dq;
	node_deform.DownloadSync(h_dq);

	average_node_se3.set_zero();
	average_node_pos = make_float3(0, 0, 0);
	float num_nodes_selected = 0;
	for (auto i = 0; i < node_list.size(); i++)
	// for (auto i = 0; i < 1; i++) // I use one for debug
	{
		average_node_se3 += h_dq[node_list[i]];
		average_node_pos += make_float3(h_node_coord[node_list[i]].x, h_node_coord[node_list[i]].y, h_node_coord[node_list[i]].z);
		num_nodes_selected += 1.f;
	}
	average_node_se3.normalize();
	auto norm_inv = 1.f / num_nodes_selected;
	average_node_pos.x = average_node_pos.x * norm_inv;
	average_node_pos.y = average_node_pos.y * norm_inv;
	average_node_pos.z = average_node_pos.z * norm_inv;
}

void star::NodeGraphManipulator::SelectNodeBySemantic(
	const GArrayView<ucharX<d_max_num_semantic>> &node_semantic_prob,
	const unsigned short semantic_id,
	const unsigned num_node_selected,
	std::vector<unsigned short> &node_list_selected)
{
	std::vector<ucharX<d_max_num_semantic>> h_node_semantic_prob;
	node_semantic_prob.Download(h_node_semantic_prob);
	std::vector<int> existed_label;
	node_list_selected.clear();
	unsigned count = 0;
	for (auto i = 0; i < h_node_semantic_prob.size(); i++)
	{
		auto node_semantic_label = max_id(h_node_semantic_prob[i]);
		// Log existed label
		if (std::find(existed_label.begin(), existed_label.end(), node_semantic_label) == existed_label.end())
			existed_label.push_back(node_semantic_label);
		if (node_semantic_label == semantic_id)
		{
			node_list_selected.push_back((unsigned short)i);
			count++;
			if (count >= num_node_selected)
				break;
		}
	}

	// Post-check
	if (count == 0)
	{
		std::cout << "Label not found!" << std::endl;
		std::cout << "Existed labels: ";
		for (auto i = 0; i < existed_label.size(); i++)
			std::cout << existed_label[i] << " ";
		std::cout << std::endl;
		throw std::exception();
	}
}