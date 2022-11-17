#include "star/warp_solver/Node2TermsIndex.h"
#include <device_launch_parameters.h>

namespace star { namespace device {

	__global__ void buildTermKeyValueKernel(
		const GArrayView<unsigned short> dense_image_knn_patch,
		const GArrayView<ushort3> node_graph,
		const unsigned node_size,
		const GArrayView<unsigned short> sparse_feature_knn_patch,
		unsigned short* __restrict__ node_keys,
		unsigned* __restrict__ term_values
	) {
		const auto term_idx = threadIdx.x + blockDim.x * blockIdx.x;

		// The offset value for term and kv
		unsigned term_offset = 0;
		unsigned kv_offset = 0;

		// This term is in the scope of dense match
		if (term_idx < (dense_image_knn_patch.Size() / d_surfel_knn_size))
		{
			const auto in_term_offset = term_idx - term_offset;
			const auto fill_offset = kv_offset + in_term_offset * d_surfel_knn_size;
#pragma unroll
			for (auto i = 0; i < d_surfel_knn_size; ++i) {
				node_keys[fill_offset + i] = dense_image_knn_patch[in_term_offset * d_surfel_knn_size + i];
				term_values[fill_offset + i] = term_idx;
			}
			return;
		}

		// For reg term
		term_offset += (dense_image_knn_patch.Size() / d_surfel_knn_size);
		kv_offset += dense_image_knn_patch.Size();
		if (term_idx < term_offset + node_graph.Size())
		{
			const auto in_term_offset = term_idx - term_offset;
			const auto fill_offset = kv_offset + 2 * in_term_offset;
			const auto node_pair = node_graph[in_term_offset];
			node_keys[fill_offset + 0] = (node_pair.x != node_pair.y) ? node_pair.x : 0xffff;
			node_keys[fill_offset + 1] = (node_pair.x != node_pair.y) ? node_pair.y : 0xffff;
			term_values[fill_offset + 0] = term_idx;
			term_values[fill_offset + 1] = term_idx;
			return;
		}

		// For node motion term
		term_offset += node_graph.Size();
		kv_offset += 2 * node_graph.Size();
		if (term_idx < term_offset + node_size)
		{
			const auto in_term_offset = term_idx - term_offset;
			const auto fill_offset = kv_offset + in_term_offset;
			node_keys[fill_offset] = in_term_offset;
			term_values[fill_offset] = term_idx;
			return;
		}

		// For sparse feature term
		term_offset += node_size;
		kv_offset += node_size;
		if (term_idx < term_offset + (sparse_feature_knn_patch.Size() / d_surfel_knn_size))
		{
			const auto in_term_offset = term_idx - term_offset;
			const auto fill_offset = kv_offset + in_term_offset * d_surfel_knn_size;
#pragma unroll
			for (auto i = 0; i < d_surfel_knn_size; ++i) {
				node_keys[fill_offset + i] = sparse_feature_knn_patch[in_term_offset * d_surfel_knn_size + i];
				term_values[fill_offset + i] = term_idx;
			}
			return;
		}
	} // The kernel to fill the key-value pairs


	__global__ void computeTermOffsetKernel(
		const GArrayView<unsigned short> sorted_term_key,
		GArraySlice<unsigned> node2term_offset,
		const unsigned node_size
	) {
		const auto idx = threadIdx.x + blockIdx.x * blockDim.x;
		if (idx >= sorted_term_key.Size()) return;

		if (idx == 0) {
			node2term_offset[0] = 0;
		}
		else {
			const auto i_0 = sorted_term_key[idx - 1];
			const auto i_1 = sorted_term_key[idx];
			if (i_0 != i_1) {
				if (i_1 != 0xffff) {
					for (auto j = i_0 + 1; j <= i_1; ++j) {
						node2term_offset[j] = idx;  // Starting from idx
					}
				}
				else {
					for (auto j = i_0 + 1; j <= node_size; ++j) {
						node2term_offset[j] = idx;  // All rest ending at idx
					}
				}
			}
			// If this is the end
			if ((idx == sorted_term_key.Size() - 1) && (i_1 != 0xffff)) {
				for (auto j = i_1 + 1; j <= node_size; ++j) {
					node2term_offset[j] = sorted_term_key.Size(); // All rest ending at idx + 1 
				}
			}
		}
	}

} // namespace device
} // namespace star


void star::Node2TermsIndex::buildTermKeyValue(cudaStream_t stream) {
	//Correct the size
	const auto num_kv_pairs = NumKeyValuePairs();
	m_node_keys.ResizeArrayOrException(num_kv_pairs);
	m_term_idx_values.ResizeArrayOrException(num_kv_pairs);

	const auto num_terms = NumTerms();
	dim3 blk(128);
	dim3 grid(divUp(num_terms, blk.x));
	device::buildTermKeyValueKernel<<<grid, blk, 0, stream>>>(
		m_term2node.dense_image_knn_patch,
		m_term2node.node_graph,
		m_term2node.node_size,
		m_term2node.sparse_feature_knn_patch,
		m_node_keys.Ptr(),
		m_term_idx_values.Ptr()
	);

	//Sync and check error
#if defined(CUDA_DEBUG_SYNC_CHECK)
	cudaSafeCall(cudaStreamSynchronize(stream));
	cudaSafeCall(cudaGetLastError());
#endif
}

void star::Node2TermsIndex::sortCompactTermIndex(cudaStream_t stream) {
	// First sort it
	m_node2term_sorter.Sort(m_node_keys.View(), m_term_idx_values.View(), stream);

	// Correct the size of nodes
	m_node2term_offset.ResizeArrayOrException(m_num_nodes + 1);
	const auto offset_slice = m_node2term_offset.Slice();
	const GArrayView<unsigned short> sorted_key_view(m_node2term_sorter.valid_sorted_key.ptr(), m_node2term_sorter.valid_sorted_key.size());

	// Compute the offset map
	dim3 blk(256);
	dim3 grid(divUp(m_node2term_sorter.valid_sorted_key.size(), blk.x));
	device::computeTermOffsetKernel<<<grid, blk, 0, stream>>>(
		sorted_key_view, offset_slice, m_num_nodes);

	// Tested with log
	//compactedIndexLog();

	//Sync and check error
#if defined(CUDA_DEBUG_SYNC_CHECK)
	cudaSafeCall(cudaStreamSynchronize(stream));
	cudaSafeCall(cudaGetLastError());
#endif
}