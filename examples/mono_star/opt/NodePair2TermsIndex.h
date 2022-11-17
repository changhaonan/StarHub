#pragma once
#include <mono_star/common/ConfigParser.h>
#include <mono_star/common/Constants.h>
#include <star/opt/solver_encode.h>
#include <star/common/macro_utils.h>
#include <star/common/common_types.h>
#include <star/common/ArrayView.h>
#include <star/common/GBufferArray.h>
#include <star/common/algorithm_types.h>
#include <mono_star/common/term_offset_types.h>
#include <memory>

namespace star
{
	class NodePair2TermsIndex
	{
	public:
		using Ptr = std::shared_ptr<NodePair2TermsIndex>;
		NodePair2TermsIndex();
		~NodePair2TermsIndex() = default;
		STAR_NO_COPY_ASSIGN(NodePair2TermsIndex);

		// Explicit allocate/de-allocate
		void AllocateBuffer();
		void ReleaseBuffer();

		// The input for index
		void SetInputs(
			GArrayView<unsigned short> dense_image_knn_patch,
			GArrayView<ushort3> node_graph,
			GArrayView<unsigned short> sparse_feature_knn_patch,
			const unsigned num_nodes);

		// The operation interface
		void BuildHalfIndex(cudaStream_t stream = 0);
		void QueryValidNodePairSize(const bool negative_exist, cudaStream_t stream = 0); // Will block the stream
		unsigned NumTerms() const;
		unsigned NumKeyValuePairs() const;

		// Build the symmetric and row index
		void BuildSymmetricAndRowBlocksIndex(cudaStream_t stream = 0);

		// The access interface
		struct NodePair2TermMap
		{
			GArrayView<unsigned> encoded_nodepair;
			GArrayView<uint2> nodepair_term_range;
			GArrayView<unsigned> nodepair_term_index;
			TermTypeOffset term_offset;
			// For bin-block csr
			GArrayView<unsigned> blkrow_offset;
			GArrayView<int> binblock_csr_rowptr;
			const int *binblock_csr_colptr;
		};
		NodePair2TermMap GetNodePair2TermMap() const;

		/* Fill the key and value given the terms
		 */
	private:
		struct
		{
			GArrayView<unsigned short> dense_image_knn_patch; // Each dense scalar term has d_surfel_knn_size nearest neighbour
			GArrayView<ushort3> node_graph;
			unsigned node_size;
			GArrayView<unsigned short> sparse_feature_knn_patch; // Same as dense feature
		} m_term2node;

		// The term offset of term2node map
		TermTypeOffset m_term_offset;
		unsigned m_num_nodes;

		/* The key-value buffer for indexing
		 */
	private:
		GBufferArray<unsigned> m_nodepair_keys;
		GBufferArray<unsigned> m_term_idx_values;

	public:
		void buildTermKeyValue(cudaStream_t stream = 0);

		/* Perform key-value sort, do compaction
		 */
	private:
		KeyValueSort<unsigned, unsigned> m_nodepair2term_sorter;
		GBufferArray<unsigned> m_segment_label;
		PrefixSum m_segment_label_prefixsum;

		// The compacted half key and values
		GBufferArray<unsigned> m_half_nodepair_keys;
		GBufferArray<unsigned> m_half_nodepair2term_offset;

	public:
		void sortCompactTermIndex(cudaStream_t stream = 0);

		/* Fill the other part of the matrix
		 */
	private:
		GBufferArray<unsigned> m_compacted_nodepair_keys;
		GBufferArray<uint2> m_nodepair_term_range;
		KeyValueSort<unsigned, uint2> m_symmetric_kv_sorter;

	public:
		void buildSymmetricCompactedIndex(cudaStream_t stream = 0);

		/* Compute the offset and length of each BLOCKED row
		 */
	private:
		GBufferArray<unsigned> m_blkrow_offset_array;
		GBufferArray<unsigned> m_blkrow_length_array;
		void blockRowOffsetSanityCheck();
		void blockRowLengthSanityCheck();

	public:
		void computeBlockRowLength(cudaStream_t stream = 0);

		/* Compute the map from block row to the elements in this row block
		 */
	private:
		GBufferArray<unsigned> m_binlength_array;
		GBufferArray<unsigned> m_binnonzeros;
		PrefixSum m_binnonzeros_prefixsum;
		GBufferArray<int> m_binblocked_csr_rowptr;
		void binLengthNonzerosSanityCheck();
		void binBlockCSRRowPtrSanityCheck();

	public:
		void computeBinLength(cudaStream_t stream = 0);
		void computeBinBlockCSRRowPtr(cudaStream_t stream = 0);

		/* Compute the column ptr for bin block csr matrix
		 */
	private:
		GBufferArray<int> m_binblocked_csr_colptr;
		void binBlockCSRColumnPtrSanityCheck();

	public:
		void nullifyBinBlockCSRColumePtr(cudaStream_t stream = 0);
		void computeBinBlockCSRColumnPtr(cudaStream_t stream = 0);

		/* Perform sanity check for nodepair2term
		 */
	public:
		void CheckHalfIndex();
		void CompactedIndexSanityCheck();
		void CompactedIndexLog();
		// Check bin size compatibility between launch configuration & global configuration
		void CheckBinSizeCompatibility();

		// Check the size and distribution of the size of index
		void IndexStatistics();

		// Check whether the reg term contains nearly all index
		// that can be exploited to implement more efficient indexing
		// Required download data and should not be used in real-time code
		void CheckRegTermIndexCompleteness();

	private:
		static void checkKNNTermIndex(int typed_term_idx,
									  const std::vector<unsigned short> &term_knn_patch,
									  unsigned encoded_nodepair);
		static void checkRegTermIndex(int smooth_term_idx, const std::vector<ushort3> &node_graph, unsigned encoded_nodepair);
	};
}