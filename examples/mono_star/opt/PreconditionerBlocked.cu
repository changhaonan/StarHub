#include <star/common/logging.h>
#include <star/common/sanity_check.h>
#include <star/opt/jacobian_utils.cuh>
#include <mono_star/opt/PreconditionerRhsBuilder.h>

void star::PreconditionerRhsBuilder::diagonalPreconditionerSanityCheck()
{
	cudaSafeCall(cudaDeviceSynchronize()); // Sync before debug
	LOG(INFO) << "Check the diagonal elements of JtJ";

	// Download the device value
	std::vector<float> diagonal_blks_dev;
	diagonal_blks_dev.resize(m_block_preconditioner.ArraySize());
	m_block_preconditioner.View().Download(diagonal_blks_dev);
	STAR_CHECK_EQ(diagonal_blks_dev.size(), d_node_variable_dim_square * (m_node2term_map.offset.Size() - 1));

	// Download the node2term map
	std::vector<unsigned> node_offset;
	std::vector<unsigned> term_index_value;
	m_node2term_map.offset.Download(node_offset);
	m_node2term_map.term_index.Download(term_index_value);

	// Check the dense depth terms
	std::vector<float> jtj_diagonal;
	jtj_diagonal.resize(diagonal_blks_dev.size());
	for (auto i = 0; i < jtj_diagonal.size(); i++)
	{
		jtj_diagonal[i] = 0.0f;
	}

	// Compute the depth term
	unsigned inspect_index = 1836;
	PenaltyConstants constants = PenaltyConstants();
	updateDenseImageJtJDiagonalHost(
		jtj_diagonal,
		m_term2jacobian_map.dense_image_term,
		constants.DenseImageSquaredVec(),
		inspect_index,
		true);
	printf("Dense: jtj_val: %f.\n", jtj_diagonal[inspect_index]);
	updateRegJtJDiagonalHost(
		jtj_diagonal,
		m_term2jacobian_map.node_graph_reg_term,
		constants.RegSquared(),
		false);
	printf("Reg: jtj_val: %f.\n", jtj_diagonal[inspect_index]);

	updateNodeTranslationJtJDiagonalHost(
		jtj_diagonal,
		m_term2jacobian_map.node_translation_term,
		constants.NodeTranslationSquared());
	printf("Node: jtj_val: %f.\n", jtj_diagonal[inspect_index]);
	// Check it
	auto relative_err = maxRelativeError(jtj_diagonal, diagonal_blks_dev, 0.01f);
	LOG(INFO) << "The relative error is " << relative_err;
}

void star::PreconditionerRhsBuilder::jacobianTransposeResidualSanityCheck()
{
	cudaSafeCall(cudaDeviceSynchronize()); // Sync before debug
	LOG(INFO) << "Check the elements of Jt Residual";

	// Compute the value at host
	const auto num_nodes = m_node2term_map.offset.Size() - 1;
	std::vector<float> jt_residual;
	jt_residual.resize(num_nodes * d_node_variable_dim);
	for (auto i = 0; i < jt_residual.size(); i++)
	{
		jt_residual[i] = 0.0f;
	}

	// Compute each terms
	PenaltyConstants constants = PenaltyConstants();
	updateDenseImageJtResidualHost(
		jt_residual,
		m_term2jacobian_map.dense_image_term,
		constants.DenseImageSquaredVec());
	updateRegJtResidualHost(
		jt_residual,
		m_term2jacobian_map.node_graph_reg_term,
		constants.RegSquared());
	updateNodeTranslationJtResidualHost(
		jt_residual,
		m_term2jacobian_map.node_translation_term,
		constants.NodeTranslationSquared());

	// Download the results from device
	std::vector<float> jt_residual_dev;
	m_jt_residual.View().Download(jt_residual_dev);
	STAR_CHECK_EQ(jt_residual.size(), jt_residual_dev.size());

	// Check it
	// auto relative_err = maxRelativeError(jt_residual, jt_residual_dev, 0.001f, true, d_node_variable_dim * 10);
	auto relative_err = maxRelativeError(jt_residual, jt_residual_dev, 0.001f);
	LOG(INFO) << "The relative error is " << relative_err;
}