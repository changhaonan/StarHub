#include "star/warp_solver/NodePair2TermsIndex.h"

star::NodePair2TermsIndex::NodePair2TermsIndex() {
	memset(&m_term2node, 0, sizeof(m_term2node));
	memset(&m_term_offset, 0, sizeof(TermTypeOffset));
}

void star::NodePair2TermsIndex::AllocateBuffer() {
	const auto& config = ConfigParser::Instance();
	const auto num_camera = config.num_cam();
	size_t num_pixels = 0;
	for (auto cam_idx = 0; cam_idx < num_camera; ++cam_idx) {
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

void star::NodePair2TermsIndex::ReleaseBuffer() {
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
	const unsigned num_nodes) {
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
		num_feature_pair
	);
}

void star::NodePair2TermsIndex::BuildHalfIndex(cudaStream_t stream) {
	buildTermKeyValue(stream);
	sortCompactTermIndex(stream);
}

void star::NodePair2TermsIndex::BuildSymmetricAndRowBlocksIndex(cudaStream_t stream) {
	buildSymmetricCompactedIndex(stream);

	// The map from row to blks
	computeBlockRowLength(stream);
	computeBinLength(stream);
	computeBinBlockCSRRowPtr(stream);

	// Compute the column ptr
	nullifyBinBlockCSRColumePtr(stream);
	computeBinBlockCSRColumnPtr(stream);
}

unsigned star::NodePair2TermsIndex::NumTerms() const {
	return
		m_term2node.dense_image_knn_patch.Size() / d_surfel_knn_size
		+ m_term2node.node_graph.Size()
		+ m_term2node.node_size
		+ m_term2node.sparse_feature_knn_patch.Size() / d_surfel_knn_size;
}

unsigned star::NodePair2TermsIndex::NumKeyValuePairs() const {
	return
		m_term2node.dense_image_knn_patch.Size() * (d_surfel_knn_size - 1) / 2
		+ m_term2node.node_graph.Size() * 1
		+ 0
		+ m_term2node.sparse_feature_knn_patch.Size() * (d_surfel_knn_size - 1) / 2;
}

star::NodePair2TermsIndex::NodePair2TermMap star::NodePair2TermsIndex::GetNodePair2TermMap() const {
	NodePair2TermMap map;
	map.encoded_nodepair = m_symmetric_kv_sorter.valid_sorted_key;
	map.nodepair_term_range = m_symmetric_kv_sorter.valid_sorted_value;
	map.nodepair_term_index = m_nodepair2term_sorter.valid_sorted_value;
	map.term_offset = m_term_offset;

	// For bin blocked csr format
	map.blkrow_offset = m_blkrow_offset_array.View();
	map.binblock_csr_rowptr = m_binblocked_csr_rowptr.View();
	map.binblock_csr_colptr = m_binblocked_csr_colptr.Ptr(); //The size is not required, and not queried
	return map;
}

void star::NodePair2TermsIndex::CheckBinSizeCompatibility() {
	// Bin Matrix size must be larger than raw matrix size
	STAR_CHECK_LT(d_max_num_nodes * d_node_variable_dim, d_bin_size* d_max_num_bin);
	// Otherwise, there is no need to use bin
	STAR_CHECK_LT(d_node_variable_dim, d_bin_size);
}

/* The method for sanity check
 */
void star::NodePair2TermsIndex::CheckHalfIndex() {
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
		for (auto j = half_offset[nodepair_idx]; j < half_offset[nodepair_idx + 1]; j++) {
			const auto term_idx = map_term_index[j];
			const auto encoded_pair = half_nodepair[nodepair_idx];
			TermType type;
			unsigned in_type_offset;
			query_typed_index(term_idx, m_term_offset, type, in_type_offset);
			switch (type) {
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

void star::NodePair2TermsIndex::CompactedIndexSanityCheck() {
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
		for (auto j = nodepair_term_range[nodepair_idx].x; j < nodepair_term_range[nodepair_idx].y; j++) {
			const auto term_idx = map_term_index[j];
			const auto encoded_pair = nodepair[nodepair_idx];
			TermType type;
			unsigned in_type_offset;
			query_typed_index(term_idx, m_term_offset, type, in_type_offset);
			switch (type) {
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
	const std::vector<unsigned short>& term_knn_patch,
	unsigned encoded_nodepair
) {
	unsigned node_i, node_j;
	decode_nodepair(encoded_nodepair, node_i, node_j);
	
	// Check node_i 
	bool matched_node_i = false;
	for (auto i = 0; i < d_surfel_knn_size; ++i) {
		if (node_i == term_knn_patch[d_surfel_knn_size * size_t(typed_term_idx) + i]) {
			matched_node_i = true;
			break;
		}
	}
	bool matched_node_j = false;
	for (auto i = 0; i < d_surfel_knn_size; ++i) {
		if (node_i == term_knn_patch[d_surfel_knn_size * size_t(typed_term_idx) + i]) {
			matched_node_j = true;
			break;
		}
	}
	if (!matched_node_i || !matched_node_j) {
		std::cout << "KNN term of " << node_i << ", " << node_j << ": ";
		for (auto i = 0; i < d_surfel_knn_size; ++i) {
			printf("%d, ", term_knn_patch[d_surfel_knn_size * size_t(typed_term_idx) + i]);
		}
		std::cout << std::endl;
	}
}

void star::NodePair2TermsIndex::checkRegTermIndex(
	int reg_term_idx,
	const std::vector<ushort3>& node_graph,
	unsigned encoded_nodepair
) {
	const auto graph_pair = node_graph[reg_term_idx];
	unsigned node_i, node_j;
	decode_nodepair(encoded_nodepair, node_i, node_j);
	auto node_idx = node_i;
	STAR_CHECK(node_idx == graph_pair.x || node_idx == graph_pair.y);
	node_idx = node_j;
	STAR_CHECK(node_idx == graph_pair.x || node_idx == graph_pair.y);
}

void star::NodePair2TermsIndex::IndexStatistics() {
	LOG(INFO) << "Performing some statistics on the index";
	const auto pair2term = GetNodePair2TermMap();
	std::vector<uint2> nodepair_term_range;
	pair2term.nodepair_term_range.Download(nodepair_term_range);

	double avg_term = 0.0;
	double min_term = 1e5;
	double max_term = 0.0;
	for (auto i = 0; i < nodepair_term_range.size(); i++) {
		uint2 term_range = nodepair_term_range[i];
		auto term_size = double(term_range.y - term_range.x);
		if (term_size < min_term) min_term = term_size;
		if (term_size > max_term) max_term = term_size;
		avg_term += term_size;
	}
	avg_term /= nodepair_term_range.size();
	LOG(INFO) << "The average size of node pair term is " << avg_term;
	LOG(INFO) << "The max size of node pair term is " << max_term;
	LOG(INFO) << "The min size of node pair term is " << min_term;
}

void star::NodePair2TermsIndex::CheckRegTermIndexCompleteness() {
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
	const TermTypeOffset& term_offset = m_term_offset;

	// Iterate over all pairs
	unsigned non_reg_pairs = 0;
	for (auto nodepair_idx = 0; nodepair_idx < nodepair.size(); nodepair_idx++) {
		// The flag variable
		bool contains_reg_term = false;

		// To iterate over all range
		const auto& term_range = nodepair_term_range[nodepair_idx];
		for (auto term_iter = term_range.x; term_iter < term_range.y; term_iter++) {
			const auto term_idx = map_term_index[term_iter];
			unsigned typed_term_idx;
			TermType term_type;
			query_typed_index(term_idx, term_offset, term_type, typed_term_idx);

			if (term_type == TermType::Reg || term_type == TermType::Feature || term_type == TermType::DenseImage) {
				contains_reg_term = true;
				break;
			}
		}

		// Increase the counter
		if (!contains_reg_term) {
			non_reg_pairs++;
		}
	}

	// Output if there is pair without reg term
	if (non_reg_pairs > 0) {
		LOG(INFO) << "There are " << non_reg_pairs << " contains no reg term of all " << nodepair.size() << " pairs!";
	}
	else {
		LOG(INFO) << "The reg term is complete";
	}
}