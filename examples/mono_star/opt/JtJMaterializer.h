#pragma once
#include <map>
#include <star/common/sanity_check.h>
#include <star/common/macro_utils.h>
#include <star/common/common_types.h>
#include <star/common/ArrayView.h>
#include <star/common/GBufferArray.h>
#include <star/common/algorithm_types.h>
#include <star/opt/solver_types.h>
#include <mono_star/common/global_configs.h>
#include <mono_star/common/ConfigParser.h>
#include <mono_star/common/Constants.h>
#include <mono_star/common/term_offset_types.h>
#include <mono_star/opt/Node2TermsIndex.h>
#include <mono_star/opt/NodePair2TermsIndex.h>
#include <mono_star/opt/PenaltyConstants.h>
#include <mono_star/opt/utils/solver_encode.h>
#include <star/pcg_solver/ApplySpMVBinBlockCSR.h>
#include <memory>

namespace star
{
	class JtJMaterializer
	{
	private:
		// The map from term to jacobian(and residual)
		Term2JacobianMaps m_term2jacobian_map;

		// The map from node to terms
		using Node2TermMap = Node2TermsIndex::Node2TermMap;
		Node2TermMap m_node2term_map;

		// The map from node pair to terms
		using NodePair2TermMap = NodePair2TermsIndex::NodePair2TermMap;
		NodePair2TermMap m_nodepair2term_map;

		// The constants value
		PenaltyConstants m_penalty_constants;

	public:
		using Ptr = std::shared_ptr<JtJMaterializer>;
		JtJMaterializer();
		~JtJMaterializer() = default;
		STAR_NO_COPY_ASSIGN(JtJMaterializer);

		// Explicit allocation, release and input
		void AllocateBuffer();
		void ReleaseBuffer();

		// The global input
		void SetInputs(
			NodePair2TermMap nodepair2term,
			DenseImageTerm2Jacobian dense_image_term,
			NodeGraphRegTerm2Jacobian node_graph_reg_term,
			NodeTranslationTerm2Jacobian node_translation_term,
			FeatureTerm2Jacobian feature_term,
			Node2TermMap node2term,
			PenaltyConstants constants = PenaltyConstants());

		// The processing method
		void BuildMaterializedJtJNondiagonalBlocks(cudaStream_t stream = 0);
		void BuildMaterializedJtJNondiagonalBlocksGlobalIteration(cudaStream_t stream = 0);

		/* The buffer for non-diagonal blocked
		 */
	private:
		GBufferArray<float> m_nondiag_blks;
		void updateRegCostJtJBlockHost(std::vector<float> &jtj_flatten, const PenaltyConstants constants = PenaltyConstants());
		void nonDiagonalBlocksSanityCheck();

	public:
		void computeNonDiagonalBlocks(cudaStream_t stream = 0);
		void computeNonDiagonalBlocksNoSync(cudaStream_t stream = 0);

		/* The method to assemble Bin-Blocked csr matrix
		 */
	private:
		GBufferArray<float> m_binblock_csr_data;
		ApplySpMVBinBlockCSR<d_node_variable_dim>::Ptr m_spmv_handler;

	public:
		void AssembleBinBlockCSR(GArrayView<float> diagonal_blks, cudaStream_t stream = 0);
		ApplySpMVBinBlockCSR<d_node_variable_dim>::Ptr GetSpMVHandler() { return m_spmv_handler; }

		/* The debug method for sparse matrix vector product
		 */
	public:
		void TestSparseMV(GArrayView<float> x, GArrayView<float> jtj_x_result);
	};

}