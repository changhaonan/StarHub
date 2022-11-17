#include "star/common/Constants.h"
#include "star/warp_solver/PreconditionerRhsBuilder.h"

void star::PreconditionerRhsBuilder::AllocateBuffer() {
	m_block_preconditioner.AllocateBuffer(d_node_variable_dim_square * Constants::kMaxNumNodes);
	m_preconditioner_inverse_handler = std::make_shared<BlockDiagonalPreconditionerInverse<d_node_variable_dim>>();
	m_preconditioner_inverse_handler->AllocateBuffer(d_node_variable_dim * Constants::kMaxNumNodes);

	m_jt_residual.AllocateBuffer(d_node_variable_dim * Constants::kMaxNumNodes);
}

void star::PreconditionerRhsBuilder::ReleaseBuffer() {
	m_block_preconditioner.ReleaseBuffer();
	m_preconditioner_inverse_handler->ReleaseBuffer();

	m_jt_residual.ReleaseBuffer();
}

void star::PreconditionerRhsBuilder::SetInputs(
	Node2TermMap node2term,
	DenseImageTerm2Jacobian dense_image_term,
	NodeGraphRegTerm2Jacobian node_graph_reg_term,
	NodeTranslationTerm2Jacobian node_translation_term,
	FeatureTerm2Jacobian feature_term,
	PenaltyConstants constants
) {
	m_node2term_map = node2term;

	m_term2jacobian_map.dense_image_term = dense_image_term;
	m_term2jacobian_map.node_graph_reg_term = node_graph_reg_term;
	m_term2jacobian_map.node_translation_term = node_translation_term;
	m_term2jacobian_map.feature_term = feature_term;
	m_penalty_constants = constants;
}


//The high level processing interface
void star::PreconditionerRhsBuilder::ComputeDiagonalPreconditioner(cudaStream_t stream) {
	ComputeDiagonalBlocks(stream);
	ComputeDiagonalPreconditionerInverse(stream);
}