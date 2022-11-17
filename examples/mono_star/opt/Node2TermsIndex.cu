#include <mono_star/opt/Node2TermsIndex.h>
#include <device_launch_parameters.h>

namespace star::device
{
	__global__ void buildTermKeyValueKernel(
		const GArrayView<unsigned short> dense_image_knn_patch,
		const GArrayView<ushort3> node_graph,
		const unsigned node_size,
		const GArrayView<unsigned short> sparse_feature_knn_patch,
		unsigned short *__restrict__ node_keys,
		unsigned *__restrict__ term_values)
	{
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
			for (auto i = 0; i < d_surfel_knn_size; ++i)
			{
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
			for (auto i = 0; i < d_surfel_knn_size; ++i)
			{
				node_keys[fill_offset + i] = sparse_feature_knn_patch[in_term_offset * d_surfel_knn_size + i];
				term_values[fill_offset + i] = term_idx;
			}
			return;
		}
	} // The kernel to fill the key-value pairs

	__global__ void computeTermOffsetKernel(
		const GArrayView<unsigned short> sorted_term_key,
		GArraySlice<unsigned> node2term_offset,
		const unsigned node_size)
	{
		const auto idx = threadIdx.x + blockIdx.x * blockDim.x;
		if (idx >= sorted_term_key.Size())
			return;

		if (idx == 0)
		{
			node2term_offset[0] = 0;
		}
		else
		{
			const auto i_0 = sorted_term_key[idx - 1];
			const auto i_1 = sorted_term_key[idx];
			if (i_0 != i_1)
			{
				if (i_1 != 0xffff)
				{
					for (auto j = i_0 + 1; j <= i_1; ++j)
					{
						node2term_offset[j] = idx; // Starting from idx
					}
				}
				else
				{
					for (auto j = i_0 + 1; j <= node_size; ++j)
					{
						node2term_offset[j] = idx; // All rest ending at idx
					}
				}
			}
			// If this is the end
			if ((idx == sorted_term_key.Size() - 1) && (i_1 != 0xffff))
			{
				for (auto j = i_1 + 1; j <= node_size; ++j)
				{
					node2term_offset[j] = sorted_term_key.Size(); // All rest ending at idx + 1
				}
			}
		}
	}
}

star::Node2TermsIndex::Node2TermsIndex()
{
	memset(&m_term2node, 0, sizeof(m_term2node));
	memset(&m_term_offset, 0, sizeof(TermTypeOffset));
	m_num_nodes = 0;
}

void star::Node2TermsIndex::AllocateBuffer()
{
	const auto &config = ConfigParser::Instance();
	const auto num_camera = config.num_cam();
	size_t num_pixels = 0;
	for (auto cam_idx = 0; cam_idx < num_camera; ++cam_idx)
	{
		num_pixels += config.downsample_img_cols(cam_idx) * config.downsample_img_rows(cam_idx);
	}
	const auto max_dense_image_terms = num_pixels;
	const auto max_node_graph_terms = Constants::kMaxNumNodes * d_node_knn_size;
	const auto max_node_translation_terms = Constants::kMaxNumNodes;
	const auto max_feature_terms = Constants::kMaxMatchedSparseFeature;

	// The total maximum size of kv buffer
	const size_t kv_buffer_size =
		max_dense_image_terms * d_surfel_knn_size +
		max_node_graph_terms * 2 +
		max_node_translation_terms +
		max_feature_terms * d_surfel_knn_size;

	// Allocate the key-value pair
	m_node_keys.AllocateBuffer(kv_buffer_size);
	m_term_idx_values.AllocateBuffer(kv_buffer_size);

	// Allocate the sorting and compaction buffer
	m_node2term_sorter.AllocateBuffer(kv_buffer_size);
	m_node2term_offset.AllocateBuffer(Constants::kMaxNumNodes + 1);
}

void star::Node2TermsIndex::ReleaseBuffer()
{
	m_node_keys.ReleaseBuffer();
	m_term_idx_values.ReleaseBuffer();
}

void star::Node2TermsIndex::SetInputs(
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

void star::Node2TermsIndex::BuildIndex(cudaStream_t stream)
{
	buildTermKeyValue(stream);
	sortCompactTermIndex(stream);

#ifdef OPT_DEBUG_CHECK
	compactedIndexSanityCheck();
#endif // OPT_DEBUG_CHECK
}

/* The size query methods
 */
unsigned star::Node2TermsIndex::NumTerms() const
{
	return m_term2node.dense_image_knn_patch.Size() / d_surfel_knn_size + m_term2node.node_graph.Size() + m_term2node.node_size + m_term2node.sparse_feature_knn_patch.Size() / d_surfel_knn_size;
}

unsigned star::Node2TermsIndex::NumKeyValuePairs() const
{
	return m_term2node.dense_image_knn_patch.Size() + m_term2node.node_graph.Size() * 2 + m_term2node.node_size + m_term2node.sparse_feature_knn_patch.Size();
}

/* A sanity check function for node2term maps
 */
void star::Node2TermsIndex::compactedIndexSanityCheck()
{
	LOG(INFO) << "Check of compacted node2term index";
	// First download the input data
	std::vector<ushort3> h_node_graph;
	std::vector<unsigned short> h_dense_image_knn_patch, h_sparse_feature_knn_patch;
	m_term2node.dense_image_knn_patch.Download(h_dense_image_knn_patch);
	m_term2node.node_graph.Download(h_node_graph);
	m_term2node.sparse_feature_knn_patch.Download(h_sparse_feature_knn_patch);

	// Next download the maps
	const auto map = GetNode2TermMap();
	std::vector<unsigned> map_offset, map_term_index;
	map.offset.Download(map_offset);
	map.term_index.Download(map_term_index);

	// Basic check
	STAR_CHECK_EQ(map_offset.size(), m_num_nodes + 1);
	STAR_CHECK_EQ(map_term_index.size(), NumKeyValuePairs());

	// Check each nodes
	for (auto node_idx = 0; node_idx < m_num_nodes; node_idx++)
	{
		for (auto j = map_offset[node_idx]; j < map_offset[size_t(node_idx) + 1]; j++)
		{
			const auto term_idx = map_term_index[j];
			TermType type;
			unsigned in_type_offset;
			query_typed_index(term_idx, map.term_offset, type, in_type_offset);
			switch (type)
			{
			case TermType::DenseImage:
				checkKnnPatchTermIndex(in_type_offset, h_dense_image_knn_patch, node_idx);
				break;
			case TermType::Reg:
				checkSmoothTermIndex(in_type_offset, h_node_graph, node_idx);
				break;
			case TermType::NodeTranslation:
				break;
			case TermType::Feature:
				checkKnnPatchTermIndex(in_type_offset, h_sparse_feature_knn_patch, node_idx);
				break;
			case TermType::Invalid:
			default:
				LOG(FATAL) << "Can not be invalid types";
			}
		}
	}

	LOG(INFO) << "Check done! Seems correct!";
}

void star::Node2TermsIndex::checkKnnPatchTermIndex(
	int typed_term_idx,
	const std::vector<unsigned short> &knn_patch_vec,
	unsigned short node_idx)
{
	bool is_neighbor = false;
	for (auto i = 0; i < d_surfel_knn_size; ++i)
	{
		const auto nn_node = knn_patch_vec[typed_term_idx * d_surfel_knn_size + i];
		if (node_idx == nn_node)
			is_neighbor = true;
	}
	STAR_CHECK(is_neighbor);
}

void star::Node2TermsIndex::checkSmoothTermIndex(int smooth_term_idx, const std::vector<ushort3> &node_graph, unsigned short node_idx)
{
	const auto node_pair = node_graph[smooth_term_idx];
	STAR_CHECK(node_idx == node_pair.x || node_idx == node_pair.y);
}

void star::Node2TermsIndex::compactedIndexLog()
{
	const auto offset_slice = m_node2term_offset.Slice();
	const GArrayView<unsigned short> sorted_key_view(m_node2term_sorter.valid_sorted_key.ptr(), m_node2term_sorter.valid_sorted_key.size());

	// Log offset
	std::vector<unsigned> h_offset;
	offset_slice.DownloadSync(h_offset);
	// for(auto i = 0; i < h_offset.size() - 1; ++i) {
	//	printf("Node:%d , term size: %d\n", i, h_offset[i + 1] - h_offset[i]);
	// }

	// Log key value
	const GArrayView<unsigned> sorted_value_view(m_node2term_sorter.valid_sorted_value.ptr(), m_node2term_sorter.valid_sorted_value.size());
	std::vector<unsigned short> h_sorted_key;
	std::vector<unsigned> h_sorted_value;
	sorted_key_view.Download(h_sorted_key);
	sorted_value_view.Download(h_sorted_value);
	for (auto i = 0; i < h_sorted_key.size(); ++i)
	{
		printf("Node: %d, Term: %d.\n", h_sorted_key[i], h_sorted_value[i]);
	}
}

void star::Node2TermsIndex::buildTermKeyValue(cudaStream_t stream)
{
	// Correct the size
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
		m_term_idx_values.Ptr());

	// Sync and check error
#if defined(CUDA_DEBUG_SYNC_CHECK)
	cudaSafeCall(cudaStreamSynchronize(stream));
	cudaSafeCall(cudaGetLastError());
#endif
}

void star::Node2TermsIndex::sortCompactTermIndex(cudaStream_t stream)
{
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
	// compactedIndexLog();

	// Sync and check error
#if defined(CUDA_DEBUG_SYNC_CHECK)
	cudaSafeCall(cudaStreamSynchronize(stream));
	cudaSafeCall(cudaGetLastError());
#endif
}