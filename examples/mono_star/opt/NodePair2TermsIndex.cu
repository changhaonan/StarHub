#include "star/common/global_configs.h"
#include "common/sanity_check.h"
#include "star/common/Macros.h"
#include "star/types/term_offset_types.h"
#include "star/warp_solver/utils/solver_encode.h"
#include "star/warp_solver/NodePair2TermsIndex.h"
#include <device_launch_parameters.h>

namespace star { namespace device {

	__host__ __device__ __forceinline__
		void computeKVNodePairKNN(
			const unsigned short* __restrict__ knn_arr,
			unsigned* __restrict__ nodepair_key
		) {
		auto offset = 0;
		for (auto i = 0; i < d_surfel_knn_size; i++) {
			const auto node_i = knn_arr[i];
			for (auto j = 0; j < d_surfel_knn_size; j++) {
				const auto node_j = knn_arr[j];
				if (node_i < node_j) {
					nodepair_key[offset] = encode_nodepair(node_i, node_j);
					offset++;
				}
			}
		}
	}


	__global__ void buildKeyValuePairKernel(
		const unsigned short* __restrict__ dense_image_knn_patch,
		const ushort3* __restrict__ node_graph,
		const unsigned node_size,
		const unsigned short* __restrict__ sparse_feature_knn_patch,
		const TermTypeOffset offset,
		unsigned* __restrict__ nodepair_keys,
		unsigned* __restrict__ term_values
	) {
		const auto term_idx = threadIdx.x + blockIdx.x * blockDim.x;
		TermType term_type;
		unsigned typed_term_idx, kv_offset;
		query_nodepair_index(term_idx, offset, term_type, typed_term_idx, kv_offset);

		// Compute the pair key locally
		unsigned term_nodepair_key[d_surfel_knn_pair_size];
		unsigned save_size = d_surfel_knn_pair_size;

		// Zero init
#pragma unroll
		for (auto i = 0; i < d_surfel_knn_pair_size; i++) {
			term_nodepair_key[i] = UNINITIALIZED_KEY;
		}

		switch (term_type) {
		case TermType::DenseImage:
			computeKVNodePairKNN(dense_image_knn_patch + d_surfel_knn_size * typed_term_idx, term_nodepair_key);
			break;
		case TermType::Reg:
			{
				const auto node_pair = node_graph[typed_term_idx];
				if (node_pair.x < node_pair.y) {
					term_nodepair_key[0] = encode_nodepair(node_pair.x, node_pair.y);
				}
				else if (node_pair.y < node_pair.x) {
					term_nodepair_key[0] = encode_nodepair(node_pair.y, node_pair.x);
				}
				save_size = 1;
			}
			break;
		case TermType::NodeTranslation:
			save_size = 0;  // No pairs here
			break;
		case TermType::Feature:
			computeKVNodePairKNN(sparse_feature_knn_patch + d_surfel_knn_size * typed_term_idx, term_nodepair_key);
			break;
		default:
			save_size = 0;
			break;
		}

		// Save it
		for (auto i = 0; i < save_size; i++) {
			nodepair_keys[kv_offset + i] = term_nodepair_key[i];
			term_values[kv_offset + i] = term_idx;
		}
	}


	__global__ void segmentNodePairKernel(
		const GArrayView<unsigned> sorted_node_pair,
		unsigned* segment_label
	) {
		// Check the valid of node size
		const auto idx = threadIdx.x + blockDim.x * blockIdx.x;
		if (idx >= sorted_node_pair.Size()) return;

		// The label must be written
		unsigned label = 0;

		// Check the size of node pair
		const auto encoded_pair = sorted_node_pair[idx];
		unsigned node_i, node_j;
		decode_nodepair(encoded_pair, node_i, node_j);
		if ((encoded_pair != UNINITIALIZED_KEY) && (node_i > d_max_num_nodes || node_j > d_max_num_nodes)) { //Take -1 into account
			// pass
		}
		else {
			if (idx == 0) label = 1;
			else // Can check the prev one
			{
				const auto encoded_prev = sorted_node_pair[idx - 1];
				if (encoded_prev != encoded_pair) label = 1;
			}
		}

		// Write to result
		segment_label[idx] = label;
	}


	__global__ void compactNodePairKeyKernel(
		const GArrayView<unsigned> sorted_node_pair,
		const unsigned* segment_label,
		const unsigned* inclusive_sum_label,
		unsigned* compacted_key,
		unsigned* compacted_offset,
		bool* negative_exist
	) {
		const auto idx = threadIdx.x + blockDim.x * blockIdx.x;
		if (idx < sorted_node_pair.Size() - 1) {
			if (segment_label[idx] > 0) {
				const auto compacted_idx = inclusive_sum_label[idx] - 1;
				compacted_key[compacted_idx] = sorted_node_pair[idx];
				compacted_offset[compacted_idx] = idx;
			}
		}
		else if (idx == sorted_node_pair.Size() - 1) {
			//The size of the sorted_key, segment label and 
			//inclusive-sumed segment are the same
			if (sorted_node_pair[idx] == UNINITIALIZED_KEY) {
				*negative_exist = true;
			}
			else {
				const auto last_idx = inclusive_sum_label[idx];
				compacted_offset[last_idx] = sorted_node_pair.Size();  // Will this be a problem?
				if (segment_label[idx] > 0) {
					const auto compacted_idx = last_idx - 1;
					compacted_key[compacted_idx] = sorted_node_pair[idx];
					compacted_offset[compacted_idx] = idx;
				}
				*negative_exist = false;
			}
		}
	}

	__global__ void computeSymmetricNodePairKernel(
		const GArrayView<unsigned> compacted_key,
		const unsigned* compacted_offset,
		unsigned* full_nodepair_key,
		uint2* full_term_start_end
	) {
		const auto idx = threadIdx.x + blockDim.x * blockIdx.x;
		if (idx < compacted_key.Size()) {
			const unsigned nodepair = compacted_key[idx];
			const unsigned start_idx = compacted_offset[idx];
			const unsigned end_idx = compacted_offset[idx + 1]; //This is safe
			unsigned node_i, node_j;
			decode_nodepair(nodepair, node_i, node_j);
			const unsigned sym_nodepair = encode_nodepair(node_j, node_i);
			//printf("i: %d, j: %d.\n", node_i, node_j);
			full_nodepair_key[2 * idx + 0] = nodepair;
			full_nodepair_key[2 * idx + 1] = sym_nodepair;
			full_term_start_end[2 * idx + 0] = make_uint2(start_idx, end_idx);
			full_term_start_end[2 * idx + 1] = make_uint2(start_idx, end_idx);
		}
	}

} // namespace device
} // namespace star

void star::NodePair2TermsIndex::buildTermKeyValue(cudaStream_t stream) {
	// Correct the size of array
	const auto num_kvs = NumKeyValuePairs();
	m_nodepair_keys.ResizeArrayOrException(num_kvs);
	m_term_idx_values.ResizeArrayOrException(num_kvs);

	const auto num_terms = NumTerms();
	dim3 blk(256);
	dim3 grid(divUp(num_terms, blk.x));
	device::buildKeyValuePairKernel<<<grid, blk, 0, stream>>>(
		m_term2node.dense_image_knn_patch.Ptr(),
		m_term2node.node_graph.Ptr(),
		m_term2node.node_size,
		m_term2node.sparse_feature_knn_patch.Ptr(),
		m_term_offset,
		m_nodepair_keys.Ptr(),
		m_term_idx_values.Ptr()
	);

	// Sync and check error
#if defined(CUDA_DEBUG_SYNC_CHECK)
	cudaSafeCall(cudaStreamSynchronize(stream));
	cudaSafeCall(cudaGetLastError());
#endif
}

void star::NodePair2TermsIndex::sortCompactTermIndex(cudaStream_t stream) {
	m_nodepair2term_sorter.Sort(m_nodepair_keys.View(), m_term_idx_values.View(), 24, stream);

	// Do segmentation
	m_segment_label.ResizeArrayOrException(m_nodepair_keys.ArraySize());
	GArrayView<unsigned> sorted_node_pair(m_nodepair2term_sorter.valid_sorted_key);
	dim3 blk(256);
	dim3 grid(divUp(sorted_node_pair.Size(), blk.x));
	device::segmentNodePairKernel<<<grid, blk, 0, stream>>>(sorted_node_pair, m_segment_label.Ptr());

	// Do prefix sum and compaction
	m_segment_label_prefixsum.InclusiveSum(m_segment_label.View(), stream);

	// Create -1 flag
	bool* negative_exist;
	cudaSafeCall(cudaMalloc((void**)&negative_exist, sizeof(bool)));
	device::compactNodePairKeyKernel<<<grid, blk, 0, stream>>>(
		sorted_node_pair,
		m_segment_label.Ptr(),
		m_segment_label_prefixsum.valid_prefixsum_array.ptr(),
		m_half_nodepair_keys.Ptr(),
		m_half_nodepair2term_offset.Ptr(),
		negative_exist
	);
	bool exist_flag;
	cudaSafeCall(cudaMemcpyAsync(&exist_flag, negative_exist, sizeof(bool), cudaMemcpyDeviceToHost, stream));

	// Sync and check error
#if defined(CUDA_DEBUG_SYNC_CHECK)
	cudaSafeCall(cudaStreamSynchronize(stream));
	cudaSafeCall(cudaGetLastError());
#endif

	QueryValidNodePairSize(exist_flag, stream); //Resize, blocking
	
	// Debug
	//CompactedIndexLog();
}

void star::NodePair2TermsIndex::buildSymmetricCompactedIndex(cudaStream_t stream) {
	//Assume the size has been queried
	dim3 blk(128);
	dim3 grid(divUp(m_half_nodepair_keys.ArraySize(), blk.x));
	device::computeSymmetricNodePairKernel<<<grid, blk, 0, stream>>>(
		m_half_nodepair_keys.View(),
		m_half_nodepair2term_offset.Ptr(),
		m_compacted_nodepair_keys.Ptr(),
		m_nodepair_term_range.Ptr()
	);

	//Sort the key-value pair
	m_symmetric_kv_sorter.Sort(m_compacted_nodepair_keys.View(), m_nodepair_term_range.View(), 24, stream);

	//Sync and check error
#if defined(CUDA_DEBUG_SYNC_CHECK)
	cudaSafeCall(cudaStreamSynchronize(stream));
	cudaSafeCall(cudaGetLastError());
#endif
}

void star::NodePair2TermsIndex::QueryValidNodePairSize(const bool negative_exist, cudaStream_t stream) {
	const unsigned* num_unique_pair_dev = m_segment_label_prefixsum.valid_prefixsum_array.ptr() + (m_segment_label_prefixsum.valid_prefixsum_array.size() - 1);
	unsigned num_unique_pair;
	cudaSafeCall(cudaMemcpyAsync(&num_unique_pair, num_unique_pair_dev, sizeof(unsigned), cudaMemcpyDeviceToHost, stream));
	cudaSafeCall(cudaStreamSynchronize(stream));

	if (negative_exist) num_unique_pair -= 1;  //Remove -1 part
	//Correct the size
	m_half_nodepair_keys.ResizeArrayOrException(num_unique_pair);
	m_half_nodepair2term_offset.ResizeArrayOrException(num_unique_pair + 1);
	m_compacted_nodepair_keys.ResizeArrayOrException(2 * num_unique_pair);
	m_nodepair_term_range.ResizeArrayOrException(2 * num_unique_pair);
}

void star::NodePair2TermsIndex::CompactedIndexLog() {
	// Check
	std::vector<unsigned> h_nodepair_keys;
	h_nodepair_keys.resize(m_nodepair2term_sorter.valid_sorted_key.size());
	std::vector<unsigned> h_term_idx_values;
	h_term_idx_values.resize(m_nodepair2term_sorter.valid_sorted_value.size());
	m_nodepair2term_sorter.valid_sorted_key.download(h_nodepair_keys.data());
	m_nodepair2term_sorter.valid_sorted_value.download(h_term_idx_values.data());

	std::vector<unsigned> h_half_nodepair_keys;
	std::vector<unsigned> h_half_nodepair2term_offset;
	m_half_nodepair_keys.View().Download(h_half_nodepair_keys);
	m_half_nodepair2term_offset.View().Download(h_half_nodepair2term_offset);

	 printf("-------------------Check----------------\n");
	 for(auto i = 0; i < h_half_nodepair_keys.size(); ++i) {
		 for(auto j = h_half_nodepair2term_offset[i]; j < h_half_nodepair2term_offset[i+1]; ++j) {
			 printf("term: %d, code: %d, offset: %d - %d\n",
				 h_term_idx_values[j], h_half_nodepair_keys[i],
				 h_half_nodepair2term_offset[i],
				 h_half_nodepair2term_offset[i+1]);
		 }
	 }

	 printf("-------------------Raw-------------------\n");
	 for(auto i = 0; i < h_term_idx_values.size(); ++i) {
		 printf("offset: %d | code: %d, term: %d\n",
			 i, h_nodepair_keys[i], h_term_idx_values[i]
		 );
	 }
	 printf("-------------------Finish---------------\n");

	//Debug
	std::vector<unsigned> sorted_key;
	m_nodepair_keys.View().Download(sorted_key);
	std::cout << sorted_key[sorted_key.size() - 1] << std::endl;
}