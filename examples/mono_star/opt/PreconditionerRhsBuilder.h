/**
 * @author Haonan Chang
 * @email chnme40cs@gmail.com
 * @create date 2022-05-04
 * @modify date 2022-05-07
 * @brief PreconditionerRhsBuilder
 */
#pragma once
#include "star/common/global_configs.h"
#include "star/warp_solver/PenaltyConstants.h"
#include "common/macro_utils.h"
#include "star/types/solver_types.h"
#include "star/warp_solver/Node2TermsIndex.h"
#include "star/warp_solver/PenaltyConstants.h"
#include "pcg_solver/BlockDiagonalPreconditionerInverse.h"
#include <memory>

namespace star {

	class PreconditionerRhsBuilder {
	private:
		// The map from term to jacobian, will also be accessed on device
		Term2JacobianMaps m_term2jacobian_map;

		// The map from node to terms
		using Node2TermMap = Node2TermsIndex::Node2TermMap;
		Node2TermMap m_node2term_map;

		// The penalty constants
		PenaltyConstants m_penalty_constants;
	public:
		using Ptr = std::shared_ptr<PreconditionerRhsBuilder>;
		STAR_DEFAULT_CONSTRUCT_DESTRUCT(PreconditionerRhsBuilder);
		STAR_NO_COPY_ASSIGN(PreconditionerRhsBuilder);

		// Explicit allocation, release and input
		void AllocateBuffer();
		void ReleaseBuffer();

		void SetInputs(
			Node2TermMap node2term,
			DenseImageTerm2Jacobian dense_image_term,
			NodeGraphRegTerm2Jacobian node_graph_reg_term,
			NodeTranslationTerm2Jacobian node_translation_term,
			FeatureTerm2Jacobian feature_term,
			PenaltyConstants constants = PenaltyConstants()
		);

		// The processing interface
		void ComputeDiagonalPreconditioner(cudaStream_t stream = 0);
		void ComputeDiagonalPreconditionerGlobalIteration(cudaStream_t stream = 0);
		GArrayView<float> InversedPreconditioner() const { return m_preconditioner_inverse_handler->InversedDiagonalBlocks(); }
		GArrayView<float> JtJDiagonalBlocks() const { return m_block_preconditioner.View(); }

		/* The buffer and method to compute the diagonal blocked pre-conditioner
		 */
	private:
		GBufferArray<float> m_block_preconditioner;

		// Methods for sanity check
		void diagonalPreconditionerSanityCheck();

		// The actual processing methods
	public:
		void ComputeDiagonalBlocks(cudaStream_t stream = 0);

		/* The buffer and method to inverse the preconditioner
		 */
	private:
		BlockDiagonalPreconditionerInverse<d_node_variable_dim>::Ptr m_preconditioner_inverse_handler;
	public:
		void ComputeDiagonalPreconditionerInverse(cudaStream_t stream = 0);
		
		/* The buffer and method to compute Jt.dot(Residual)
		 */
	private:
		GBufferArray<float> m_jt_residual;

		// Methods for sanity check
		void jacobianTransposeResidualSanityCheck();
		
	public:
		void ComputeJtResidualIndexed(cudaStream_t stream = 0);
		void ComputeJtResidual(cudaStream_t stream = 0);
		void ComputeJtResidualGlobalIteration(cudaStream_t stream = 0);
		void ComputeJtResidualLocalIteration(cudaStream_t stream = 0);
		GArrayView<float> JtDotResidualValue() const { return m_jt_residual.View(); }
	};

}