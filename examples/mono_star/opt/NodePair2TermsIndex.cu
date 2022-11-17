#include <star/common/sanity_check.h>
#include <star/opt/solver_encode.h>
#include <mono_star/common/global_configs.h>
#include <mono_star/common/Macros.h>
#include <mono_star/common/term_offset_types.h>
#include <mono_star/opt/NodePair2TermsIndex.h>
#include <device_launch_parameters.h>

namespace star::device
{
	__host__ __device__ __forceinline__ void computeKVNodePairKNN(
		const unsigned short *__restrict__ knn_arr,
		unsigned *__restrict__ nodepair_key)
	{
		auto offset = 0;
		for (auto i = 0; i < d_surfel_knn_size; i++)
		{
			const auto node_i = knn_arr[i];
			for (auto j = 0; j < d_surfel_knn_size; j++)
			{
				const auto node_j = knn_arr[j];
				if (node_i < node_j)
				{
					nodepair_key[offset] = encode_nodepair(node_i, node_j);
					offset++;
				}
			}
		}
	}

	__global__ void buildKeyValuePairKernel(
		const unsigned short *__restrict__ dense_image_knn_patch,
		const ushort3 *__restrict__ node_graph,
		const unsigned node_size,
		const unsigned short *__restrict__ sparse_feature_knn_patch,
		const TermTypeOffset offset,
		unsigned *__restrict__ nodepair_keys,
		unsigned *__restrict__ term_values)
	{
		const auto term_idx = threadIdx.x + blockIdx.x * blockDim.x;
		TermType term_type;
		unsigned typed_term_idx, kv_offset;
		query_nodepair_index(term_idx, offset, term_type, typed_term_idx, kv_offset);

		// Compute the pair key locally
		unsigned term_nodepair_key[d_surfel_knn_pair_size];
		unsigned save_size = d_surfel_knn_pair_size;

		// Zero init
#pragma unroll
		for (auto i = 0; i < d_surfel_knn_pair_size; i++)
		{
			term_nodepair_key[i] = UNINITIALIZED_KEY;
		}

		switch (term_type)
		{
		case TermType::DenseImage:
			computeKVNodePairKNN(dense_image_knn_patch + d_surfel_knn_size * typed_term_idx, term_nodepair_key);
			break;
		case TermType::Reg:
		{
			const auto node_pair = node_graph[typed_term_idx];
			if (node_pair.x < node_pair.y)
			{
				term_nodepair_key[0] = encode_nodepair(node_pair.x, node_pair.y);
			}
			else if (node_pair.y < node_pair.x)
			{
				term_nodepair_key[0] = encode_nodepair(node_pair.y, node_pair.x);
			}
			save_size = 1;
		}
		break;
		case TermType::NodeTranslation:
			save_size = 0; // No pairs here
			break;
		case TermType::Feature:
			computeKVNodePairKNN(sparse_feature_knn_patch + d_surfel_knn_size * typed_term_idx, term_nodepair_key);
			break;
		default:
			save_size = 0;
			break;
		}

		// Save it
		for (auto i = 0; i < save_size; i++)
		{
			nodepair_keys[kv_offset + i] = term_nodepair_key[i];
			term_values[kv_offset + i] = term_idx;
		}
	}

	__global__ void segmentNodePairKernel(
		const GArrayView<unsigned> sorted_node_pair,
		unsigned *segment_label)
	{
		// Check the valid of node size
		const auto idx = threadIdx.x + blockDim.x * blockIdx.x;
		if (idx >= sorted_node_pair.Size())
			return;

		// The label must be written
		unsigned label = 0;

		// Check the size of node pair
		const auto encoded_pair = sorted_node_pair[idx];
		unsigned node_i, node_j;
		decode_nodepair(encoded_pair, node_i, node_j);
		if ((encoded_pair != UNINITIALIZED_KEY) && (node_i > d_max_num_nodes || node_j > d_max_num_nodes))
		{ // Take -1 into account
		  // pass
		}
		else
		{
			if (idx == 0)
				label = 1;
			else // Can check the prev one
			{
				const auto encoded_prev = sorted_node_pair[idx - 1];
				if (encoded_prev != encoded_pair)
					label = 1;
			}
		}

		// Write to result
		segment_label[idx] = label;
	}

	__global__ void compactNodePairKeyKernel(
		const GArrayView<unsigned> sorted_node_pair,
		const unsigned *segment_label,
		const unsigned *inclusive_sum_label,
		unsigned *compacted_key,
		unsigned *compacted_offset,
		bool *negative_exist)
	{
		const auto idx = threadIdx.x + blockDim.x * blockIdx.x;
		if (idx < sorted_node_pair.Size() - 1)
		{
			if (segment_label[idx] > 0)
			{
				const auto compacted_idx = inclusive_sum_label[idx] - 1;
				compacted_key[compacted_idx] = sorted_node_pair[idx];
				compacted_offset[compacted_idx] = idx;
			}
		}
		else if (idx == sorted_node_pair.Size() - 1)
		{
			// The size of the sorted_key, segment label and
			// inclusive-sumed segment are the same
			if (sorted_node_pair[idx] == UNINITIALIZED_KEY)
			{
				*negative_exist = true;
			}
			else
			{
				const auto last_idx = inclusive_sum_label[idx];
				compacted_offset[last_idx] = sorted_node_pair.Size(); // Will this be a problem?
				if (segment_label[idx] > 0)
				{
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
		const unsigned *compacted_offset,
		unsigned *full_nodepair_key,
		uint2 *full_term_start_end)
	{
		const auto idx = threadIdx.x + blockDim.x * blockIdx.x;
		if (idx < compacted_key.Size())
		{
			const unsigned nodepair = compacted_key[idx];
			const unsigned start_idx = compacted_offset[idx];
			const unsigned end_idx = compacted_offset[idx + 1]; // This is safe
			unsigned node_i, node_j;
			decode_nodepair(nodepair, node_i, node_j);
			const unsigned sym_nodepair = encode_nodepair(node_j, node_i);
			// printf("i: %d, j: %d.\n", node_i, node_j);
			full_nodepair_key[2 * idx + 0] = nodepair;
			full_nodepair_key[2 * idx + 1] = sym_nodepair;
			full_term_start_end[2 * idx + 0] = make_uint2(start_idx, end_idx);
			full_term_start_end[2 * idx + 1] = make_uint2(start_idx, end_idx);
		}
	}
}

star::NodePair2TermsIndex::NodePair2TermsIndex()
{
	memset(&m_term2node, 0, sizeof(m_term2node));
	memset(&m_term_offset, 0, sizeof(TermTypeOffset));
}

void star::NodePair2TermsIndex::AllocateBuffer()
{
	const auto &config = ConfigParser::Instance();
	const auto num_camera = config.num_cam();
	size_t num_pixels = 0;
	for (auto cam_idx = 0; cam_idx < num_camera; ++cam_idx)
	{
		num_pixels += size_t(config.downsample_img_cols(cam_idx)) * size_t(config.downsample_img_rows(cam_idx));
	}
	const auto max_dense_image_terms = num_pixels;
	const auto max_node_graph_terms = Constants::kMaxNumNodes * d_node_knn_size;
	const auto max_node_translation_terms = Constants::kMaxNumNodes;
	const auto max_feature_terms = Constants::kMaxMatchedSparseFeature;
	const size_t kv_buffer_size =
		size_t(max_dense_image_terms) * TermPairSize::DenseImage +
		size_t(max_node_graph_terms) * TermPairSize::Reg +
		size_t(max_node_translation_terms) * TermPairSize::NodeTranslation +
		size_t(max_feature_terms) * TermPairSize::Feature;

	// Allocate the key-value pair
	m_nodepair_keys.AllocateBuffer(kv_buffer_size);
	m_term_idx_values.AllocateBuffer(kv_buffer_size);

	// Allocate the sorter and compaction
	m_nodepair2term_sorter.AllocateBuffer(kv_buffer_size);
	m_segment_label.AllocateBuffer(kv_buffer_size);
	m_segment_label_prefixsum.AllocateBuffer(kv_buffer_size);
	const auto max_unique_nodepair = Constants::kMaxNumNodePairs;
	m_half_nodepair_keys.AllocateBuffer(max_unique_nodepair);
	m_half_nodepair2term_offset.AllocateBuffer(max_unique_nodepair);

	// The buffer for symmetric index
	m_compacted_nodepair_keys.AllocateBuffer(2 * size_t(max_unique_nodepair));
	m_nodepair_term_range.AllocateBuffer(2 * size_t(max_unique_nodepair));
	m_symmetric_kv_sorter.AllocateBuffer(2 * size_t(max_unique_nodepair));

	// For blocked offset and length of each row
	m_blkrow_offset_array.AllocateBuffer(size_t(Constants::kMaxNumNodes) + 1);
	m_blkrow_length_array.AllocateBuffer(Constants::kMaxNumNodes);

	// For offset measured in bin
	const auto max_bins = divUp(Constants::kMaxNumNodes * d_node_variable_dim, d_bin_size);
	m_binlength_array.AllocateBuffer(max_bins);
	m_binnonzeros.AllocateBuffer(size_t(max_bins) + 1);
	m_binnonzeros_prefixsum.AllocateBuffer(size_t(max_bins) + 1);
	m_binblocked_csr_rowptr.AllocateBuffer(size_t(d_bin_size) * (size_t(max_bins) + 1));

	// For the colptr of bin blocked csr format
	m_binblocked_csr_colptr.AllocateBuffer(size_t(Constants::kMaxNumNodePairs) * d_node_variable_dim);

	// Check compatiblity
	CheckBinSizeCompatibility();
}

void star::NodePair2TermsIndex::ReleaseBuffer()
{
	m_nodepair_keys.ReleaseBuffer();
	m_term_idx_values.ReleaseBuffer();

	m_segment_label.ReleaseBuffer();

	m_compacted_nodepair_keys.ReleaseBuffer();
	m_nodepair_term_range.ReleaseBuffer();

	m_binlength_array.ReleaseBuffer();
	m_binnonzeros.ReleaseBuffer();
	m_binblocked_csr_rowptr.ReleaseBuffer();
}

void star::NodePair2TermsIndex::SetInputs(
	GArrayView<unsigned short> dense_image_knn_patch,
	GArrayView<ushort3> node_graph,
	GArrayView<unsigned short> sparse_feature_knn_patch,
	const unsigned num_nodes)
{
	m_term2node.dense_image_knn_patch = dense_image_knn_patch;
	m_term2node.node_graph = node_graph;
	m_term2node.node_size = num_nodes;
	m_term2node.sparse_feature_knn_patch = sparse_feature_knn_patch;
	m_num_nodes = num_nodes;

	unsigned num_dense_pixel_pair = dense_image_knn_patch.Size() / d_surfel_knn_size;
	unsigned num_node_graph_pair = node_graph.Size();
	unsigned num_feature_pair = sparse_feature_knn_patch.Size() / d_surfel_knn_size;
	// build the offset of these terms
	size2offset(
		m_term_offset,
		num_dense_pixel_pair,
		num_node_graph_pair,
		num_nodes,
		num_feature_pair);
}

void star::NodePair2TermsIndex::BuildHalfIndex(cudaStream_t stream)
{
	buildTermKeyValue(stream);
	sortCompactTermIndex(stream);
}

void star::NodePair2TermsIndex::BuildSymmetricAndRowBlocksIndex(cudaStream_t stream)
{
	buildSymmetricCompactedIndex(stream);

	// The map from row to blks
	computeBlockRowLength(stream);
	computeBinLength(stream);
	computeBinBlockCSRRowPtr(stream);

	// Compute the column ptr
	nullifyBinBlockCSRColumePtr(stream);
	computeBinBlockCSRColumnPtr(stream);
}

unsigned star::NodePair2TermsIndex::NumTerms() const
{
	return m_term2node.dense_image_knn_patch.Size() / d_surfel_knn_size + m_term2node.node_graph.Size() + m_term2node.node_size + m_term2node.sparse_feature_knn_patch.Size() / d_surfel_knn_size;
}

unsigned star::NodePair2TermsIndex::NumKeyValuePairs() const
{
	return m_term2node.dense_image_knn_patch.Size() * (d_surfel_knn_size - 1) / 2 + m_term2node.node_graph.Size() * 1 + 0 + m_term2node.sparse_feature_knn_patch.Size() * (d_surfel_knn_size - 1) / 2;
}

star::NodePair2TermsIndex::NodePair2TermMap star::NodePair2TermsIndex::GetNodePair2TermMap() const
{
	NodePair2TermMap map;
	map.encoded_nodepair = m_symmetric_kv_sorter.valid_sorted_key;
	map.nodepair_term_range = m_symmetric_kv_sorter.valid_sorted_value;
	map.nodepair_term_index = m_nodepair2term_sorter.valid_sorted_value;
	map.term_offset = m_term_offset;

	// For bin blocked csr format
	map.blkrow_offset = m_blkrow_offset_array.View();
	map.binblock_csr_rowptr = m_binblocked_csr_rowptr.View();
	map.binblock_csr_colptr = m_binblocked_csr_colptr.Ptr(); // The size is not required, and not queried
	return map;
}

void star::NodePair2TermsIndex::CheckBinSizeCompatibility()
{
	// Bin Matrix size must be larger than raw matrix size
	STAR_CHECK_LT(d_max_num_nodes * d_node_variable_dim, d_bin_size * d_max_num_bin);
	// Otherwise, there is no need to use bin
	STAR_CHECK_LT(d_node_variable_dim, d_bin_size);
}

/* The method for sanity check
 */
void star::NodePair2TermsIndex::CheckHalfIndex()
{
	LOG(INFO) << "Sanity check of the pair2term map half index";

	// First download the input data
	std::vector<ushort3> h_node_graph;
	std::vector<unsigned short> h_dense_image_knn_patch, h_feature_knn_patch, h_sparse_feature_knn_patch;
	m_term2node.dense_image_knn_patch.Download(h_dense_image_knn_patch);
	m_term2node.node_graph.Download(h_node_graph);
	m_term2node.sparse_feature_knn_patch.Download(h_sparse_feature_knn_patch);

	// Next download the value
	std::vector<unsigned> half_nodepair, half_offset, map_term_index;
	m_half_nodepair_keys.View().Download(half_nodepair);
	m_half_nodepair2term_offset.View().Download(half_offset);
	m_nodepair2term_sorter.valid_sorted_value.download(map_term_index);

	// Iterate over pairs
	printf("----------- Checking -------------\n");
	for (auto nodepair_idx = 0; nodepair_idx < half_nodepair.size(); nodepair_idx++)
	{
		for (auto j = half_offset[nodepair_idx]; j < half_offset[nodepair_idx + 1]; j++)
		{
			const auto term_idx = map_term_index[j];
			const auto encoded_pair = half_nodepair[nodepair_idx];
			TermType type;
			unsigned in_type_offset;
			query_typed_index(term_idx, m_term_offset, type, in_type_offset);
			switch (type)
			{
			case TermType::DenseImage:
				checkKNNTermIndex(in_type_offset, h_dense_image_knn_patch, encoded_pair);
				break;
			case TermType::Reg:
				checkRegTermIndex(in_type_offset, h_node_graph, encoded_pair);
				break;
			case TermType::NodeTranslation:
				break;
			case TermType::Feature:
				checkKNNTermIndex(in_type_offset, h_sparse_feature_knn_patch, encoded_pair);
				break;
			case TermType::Invalid:
			default:
				LOG(FATAL) << "Can not be invalid types";
			}
		}
	}

	LOG(INFO) << "Check done";
}

void star::NodePair2TermsIndex::CompactedIndexSanityCheck()
{
	LOG(INFO) << "Sanity check of the pair2term map";

	// First download the input data
	std::vector<ushort3> h_node_graph;
	std::vector<unsigned short> h_dense_image_knn_patch, h_feature_knn_patch, h_sparse_feature_knn_patch;
	m_term2node.dense_image_knn_patch.Download(h_dense_image_knn_patch);
	m_term2node.node_graph.Download(h_node_graph);
	m_term2node.sparse_feature_knn_patch.Download(h_sparse_feature_knn_patch);

	// Download the required map data
	const auto pair2term = GetNodePair2TermMap();
	std::vector<unsigned> nodepair, map_term_index;
	std::vector<uint2> nodepair_term_range;
	pair2term.encoded_nodepair.Download(nodepair);
	pair2term.nodepair_term_index.Download(map_term_index);
	pair2term.nodepair_term_range.Download(nodepair_term_range);

	// Basic check
	STAR_CHECK_EQ(map_term_index.size(), NumKeyValuePairs());

	// Check each nodes
	for (auto nodepair_idx = 0; nodepair_idx < nodepair.size(); nodepair_idx++)
	{
		for (auto j = nodepair_term_range[nodepair_idx].x; j < nodepair_term_range[nodepair_idx].y; j++)
		{
			const auto term_idx = map_term_index[j];
			const auto encoded_pair = nodepair[nodepair_idx];
			TermType type;
			unsigned in_type_offset;
			query_typed_index(term_idx, m_term_offset, type, in_type_offset);
			switch (type)
			{
			case TermType::DenseImage:
				checkKNNTermIndex(in_type_offset, h_dense_image_knn_patch, encoded_pair);
				break;
			case TermType::Reg:
				checkRegTermIndex(in_type_offset, h_node_graph, encoded_pair);
				break;
			case TermType::NodeTranslation:
				break;
			case TermType::Feature:
				checkKNNTermIndex(in_type_offset, h_sparse_feature_knn_patch, encoded_pair);
				break;
			case TermType::Invalid:
			default:
				LOG(FATAL) << "Can not be invalid types";
			}
		}
	}

	LOG(INFO) << "Check done";
}

void star::NodePair2TermsIndex::checkKNNTermIndex(
	int typed_term_idx,
	const std::vector<unsigned short> &term_knn_patch,
	unsigned encoded_nodepair)
{
	unsigned node_i, node_j;
	decode_nodepair(encoded_nodepair, node_i, node_j);

	// Check node_i
	bool matched_node_i = false;
	for (auto i = 0; i < d_surfel_knn_size; ++i)
	{
		if (node_i == term_knn_patch[d_surfel_knn_size * size_t(typed_term_idx) + i])
		{
			matched_node_i = true;
			break;
		}
	}
	bool matched_node_j = false;
	for (auto i = 0; i < d_surfel_knn_size; ++i)
	{
		if (node_i == term_knn_patch[d_surfel_knn_size * size_t(typed_term_idx) + i])
		{
			matched_node_j = true;
			break;
		}
	}
	if (!matched_node_i || !matched_node_j)
	{
		std::cout << "KNN term of " << node_i << ", " << node_j << ": ";
		for (auto i = 0; i < d_surfel_knn_size; ++i)
		{
			printf("%d, ", term_knn_patch[d_surfel_knn_size * size_t(typed_term_idx) + i]);
		}
		std::cout << std::endl;
	}
}

void star::NodePair2TermsIndex::checkRegTermIndex(
	int reg_term_idx,
	const std::vector<ushort3> &node_graph,
	unsigned encoded_nodepair)
{
	const auto graph_pair = node_graph[reg_term_idx];
	unsigned node_i, node_j;
	decode_nodepair(encoded_nodepair, node_i, node_j);
	auto node_idx = node_i;
	STAR_CHECK(node_idx == graph_pair.x || node_idx == graph_pair.y);
	node_idx = node_j;
	STAR_CHECK(node_idx == graph_pair.x || node_idx == graph_pair.y);
}

void star::NodePair2TermsIndex::IndexStatistics()
{
	LOG(INFO) << "Performing some statistics on the index";
	const auto pair2term = GetNodePair2TermMap();
	std::vector<uint2> nodepair_term_range;
	pair2term.nodepair_term_range.Download(nodepair_term_range);

	double avg_term = 0.0;
	double min_term = 1e5;
	double max_term = 0.0;
	for (auto i = 0; i < nodepair_term_range.size(); i++)
	{
		uint2 term_range = nodepair_term_range[i];
		auto term_size = double(term_range.y - term_range.x);
		if (term_size < min_term)
			min_term = term_size;
		if (term_size > max_term)
			max_term = term_size;
		avg_term += term_size;
	}
	avg_term /= nodepair_term_range.size();
	LOG(INFO) << "The average size of node pair term is " << avg_term;
	LOG(INFO) << "The max size of node pair term is " << max_term;
	LOG(INFO) << "The min size of node pair term is " << min_term;
}

void star::NodePair2TermsIndex::CheckRegTermIndexCompleteness()
{
	// Download the required map data
	const auto pair2term = GetNodePair2TermMap();
	std::vector<unsigned> nodepair, map_term_index;
	std::vector<uint2> nodepair_term_range;
	pair2term.encoded_nodepair.Download(nodepair);
	pair2term.nodepair_term_index.Download(map_term_index);
	pair2term.nodepair_term_range.Download(nodepair_term_range);

	// Basic check
	STAR_CHECK_EQ(map_term_index.size(), NumKeyValuePairs());
	STAR_CHECK_EQ(nodepair.size(), nodepair_term_range.size());

	// The global term offset
	const TermTypeOffset &term_offset = m_term_offset;

	// Iterate over all pairs
	unsigned non_reg_pairs = 0;
	for (auto nodepair_idx = 0; nodepair_idx < nodepair.size(); nodepair_idx++)
	{
		// The flag variable
		bool contains_reg_term = false;

		// To iterate over all range
		const auto &term_range = nodepair_term_range[nodepair_idx];
		for (auto term_iter = term_range.x; term_iter < term_range.y; term_iter++)
		{
			const auto term_idx = map_term_index[term_iter];
			unsigned typed_term_idx;
			TermType term_type;
			query_typed_index(term_idx, term_offset, term_type, typed_term_idx);

			if (term_type == TermType::Reg || term_type == TermType::Feature || term_type == TermType::DenseImage)
			{
				contains_reg_term = true;
				break;
			}
		}

		// Increase the counter
		if (!contains_reg_term)
		{
			non_reg_pairs++;
		}
	}

	// Output if there is pair without reg term
	if (non_reg_pairs > 0)
	{
		LOG(INFO) << "There are " << non_reg_pairs << " contains no reg term of all " << nodepair.size() << " pairs!";
	}
	else
	{
		LOG(INFO) << "The reg term is complete";
	}
}

void star::NodePair2TermsIndex::buildTermKeyValue(cudaStream_t stream)
{
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
		m_term_idx_values.Ptr());

	// Sync and check error
#if defined(CUDA_DEBUG_SYNC_CHECK)
	cudaSafeCall(cudaStreamSynchronize(stream));
	cudaSafeCall(cudaGetLastError());
#endif
}

void star::NodePair2TermsIndex::sortCompactTermIndex(cudaStream_t stream)
{
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
	bool *negative_exist;
	cudaSafeCall(cudaMalloc((void **)&negative_exist, sizeof(bool)));
	device::compactNodePairKeyKernel<<<grid, blk, 0, stream>>>(
		sorted_node_pair,
		m_segment_label.Ptr(),
		m_segment_label_prefixsum.valid_prefixsum_array.ptr(),
		m_half_nodepair_keys.Ptr(),
		m_half_nodepair2term_offset.Ptr(),
		negative_exist);
	bool exist_flag;
	cudaSafeCall(cudaMemcpyAsync(&exist_flag, negative_exist, sizeof(bool), cudaMemcpyDeviceToHost, stream));

	// Sync and check error
#if defined(CUDA_DEBUG_SYNC_CHECK)
	cudaSafeCall(cudaStreamSynchronize(stream));
	cudaSafeCall(cudaGetLastError());
#endif

	QueryValidNodePairSize(exist_flag, stream); // Resize, blocking

	// Debug
	// CompactedIndexLog();
}

void star::NodePair2TermsIndex::buildSymmetricCompactedIndex(cudaStream_t stream)
{
	// Assume the size has been queried
	dim3 blk(128);
	dim3 grid(divUp(m_half_nodepair_keys.ArraySize(), blk.x));
	device::computeSymmetricNodePairKernel<<<grid, blk, 0, stream>>>(
		m_half_nodepair_keys.View(),
		m_half_nodepair2term_offset.Ptr(),
		m_compacted_nodepair_keys.Ptr(),
		m_nodepair_term_range.Ptr());

	// Sort the key-value pair
	m_symmetric_kv_sorter.Sort(m_compacted_nodepair_keys.View(), m_nodepair_term_range.View(), 24, stream);

	// Sync and check error
#if defined(CUDA_DEBUG_SYNC_CHECK)
	cudaSafeCall(cudaStreamSynchronize(stream));
	cudaSafeCall(cudaGetLastError());
#endif
}

void star::NodePair2TermsIndex::QueryValidNodePairSize(const bool negative_exist, cudaStream_t stream)
{
	const unsigned *num_unique_pair_dev = m_segment_label_prefixsum.valid_prefixsum_array.ptr() + (m_segment_label_prefixsum.valid_prefixsum_array.size() - 1);
	unsigned num_unique_pair;
	cudaSafeCall(cudaMemcpyAsync(&num_unique_pair, num_unique_pair_dev, sizeof(unsigned), cudaMemcpyDeviceToHost, stream));
	cudaSafeCall(cudaStreamSynchronize(stream));

	if (negative_exist)
		num_unique_pair -= 1; // Remove -1 part
	// Correct the size
	m_half_nodepair_keys.ResizeArrayOrException(num_unique_pair);
	m_half_nodepair2term_offset.ResizeArrayOrException(num_unique_pair + 1);
	m_compacted_nodepair_keys.ResizeArrayOrException(2 * num_unique_pair);
	m_nodepair_term_range.ResizeArrayOrException(2 * num_unique_pair);
}

void star::NodePair2TermsIndex::CompactedIndexLog()
{
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
	for (auto i = 0; i < h_half_nodepair_keys.size(); ++i)
	{
		for (auto j = h_half_nodepair2term_offset[i]; j < h_half_nodepair2term_offset[i + 1]; ++j)
		{
			printf("term: %d, code: %d, offset: %d - %d\n",
				   h_term_idx_values[j], h_half_nodepair_keys[i],
				   h_half_nodepair2term_offset[i],
				   h_half_nodepair2term_offset[i + 1]);
		}
	}

	printf("-------------------Raw-------------------\n");
	for (auto i = 0; i < h_term_idx_values.size(); ++i)
	{
		printf("offset: %d | code: %d, term: %d\n",
			   i, h_nodepair_keys[i], h_term_idx_values[i]);
	}
	printf("-------------------Finish---------------\n");

	// Debug
	std::vector<unsigned> sorted_key;
	m_nodepair_keys.View().Download(sorted_key);
	std::cout << sorted_key[sorted_key.size() - 1] << std::endl;
}