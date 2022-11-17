#pragma once
#include <star/common/macro_utils.h>
#include <star/common/common_types.h>
#include <star/common/ArrayView.h>
#include <star/common/GBufferArray.h>
#include <star/common/algorithm_types.h>
#include <mono_star/common/term_offset_types.h>
#include <mono_star/common/ConfigParser.h>
#include <mono_star/common/Constants.h>
#include <memory>

namespace star
{
	class Node2TermsIndex
	{
	private:
		// Knn map
		struct
		{
			GArrayView<unsigned short> dense_image_knn_patch; // Each dense scalar term has d_surfel_knn_size nearest neighbour
			GArrayView<ushort3> node_graph;
			unsigned node_size;
			GArrayView<unsigned short> sparse_feature_knn_patch; // Same as dense feature
		} m_term2node;
		unsigned m_num_nodes;

		// The term offset of term2node map
		TermTypeOffset m_term_offset;

	public:
		using Ptr = std::shared_ptr<Node2TermsIndex>;
		Node2TermsIndex();
		~Node2TermsIndex() = default;
		STAR_NO_COPY_ASSIGN_MOVE(Node2TermsIndex);
		void AllocateBuffer();
		void ReleaseBuffer();

	public:
		void SetInputs(
			GArrayView<unsigned short> dense_image_knn_patch,
			GArrayView<ushort3> node_graph,
			GArrayView<unsigned short> sparse_feature_knn_patch,
			const unsigned num_nodes);

		// The main interface
		void BuildIndex(cudaStream_t stream = 0);
		unsigned NumTerms() const;
		unsigned NumKeyValuePairs() const;

		/* Fill the key and value given the terms
		 */
	private:
		GBufferArray<unsigned short> m_node_keys;
		GBufferArray<unsigned> m_term_idx_values;

	public:
		void buildTermKeyValue(cudaStream_t stream = 0);

		/* Perform key-value sort, do compaction
		 */
	private:
		KeyValueSort<unsigned short, unsigned> m_node2term_sorter;
		GBufferArray<unsigned> m_node2term_offset;

	public:
		void sortCompactTermIndex(cudaStream_t stream = 0);

		/* A series of checking functions
		 */
	private:
		static void checkKnnPatchTermIndex(int typed_term_idx, const std::vector<unsigned short> &knn_patch_vec, unsigned short node_idx);
		static void checkSmoothTermIndex(int smooth_term_idx, const std::vector<ushort3> &node_graph, unsigned short node_idx);
		void compactedIndexSanityCheck();
		void compactedIndexLog();
		/* The accessing interface
		 * Depends on BuildIndex
		 */
	public:
		struct Node2TermMap
		{
			GArrayView<unsigned> offset;
			GArrayView<unsigned> term_index;
			TermTypeOffset term_offset;
		};

		// Return the outside-accessed index
		Node2TermMap GetNode2TermMap() const
		{
			Node2TermMap map;
			map.offset = m_node2term_offset.View();
			map.term_index = m_node2term_sorter.valid_sorted_value;
			map.term_offset = m_term_offset;
			return map;
		}
	};
}