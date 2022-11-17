#include <star/common/sanity_check.h>
#include <star/opt/jacobian_utils.cuh>
#include <star/pcg_solver/ApplySpMVBinBlockCSR.h>
#include <mono_star/opt/JtJMaterializer.h>
#include <device_launch_parameters.h>

void star::JtJMaterializer::nonDiagonalBlocksSanityCheck()
{
	LOG(INFO) << "Sanity check of materialized non-diagonal JtJ block";

	// 1 - Prepare
	std::vector<float> jtj_blocks;
	const auto num_nodepairs = m_nodepair2term_map.nodepair_term_range.Size();
	jtj_blocks.resize(d_node_variable_dim_square * num_nodepairs);
	memset(jtj_blocks.data(), 0, sizeof(float) * num_nodepairs * d_node_variable_dim_square);
	PenaltyConstants constants = PenaltyConstants();
	// 2 - Update DenseImage Jacobain
	updateDenseImageJtJBlockHost(
		m_nodepair2term_map.encoded_nodepair,
		jtj_blocks,
		m_term2jacobian_map.dense_image_term,
		constants.DenseImageSquaredVec());
	// 3 - Upate Reg Jacobian
	updateRegJtJBlockHost(
		m_nodepair2term_map.encoded_nodepair,
		jtj_blocks,
		m_term2jacobian_map.node_graph_reg_term,
		constants.RegSquared());
	// 4 - Update NodeTranslation Jacobian
	// 5 - Update Feature Jacobian
	// FIXEME: to add
	// 6 - Download the data from device
	std::vector<float> jtj_blocks_dev;
	m_nondiag_blks.View().Download(jtj_blocks_dev);
	STAR_CHECK_EQ(jtj_blocks.size(), jtj_blocks_dev.size());
	// 7 - Compute the error
	auto relative_err = maxRelativeError(jtj_blocks, jtj_blocks_dev, 1e-3f);
	// for(auto i = 0; i < 10 * d_node_variable_dim; i++) {
	//	auto dev_value = jtj_blocks_dev[i];
	//	auto host_value = jtj_blocks[i];
	//	printf("dev: %f, host: %f.\n", dev_value, host_value);
	// }
	LOG(INFO) << "The relative error for non-diagonal jtj blocks is " << relative_err;
}

// An integrated test on the correctness of spmv
void star::JtJMaterializer::TestSparseMV(
	GArrayView<float> x,
	GArrayView<float> jtj_x_result)
{
	LOG(INFO) << "Check materialized spmv given input groundtruth";
	// Construct the matrix applier
	ApplySpMVBinBlockCSR<d_node_variable_dim> spmv_handler;
	spmv_handler.SetInputs(
		m_binblock_csr_data.Ptr(),
		m_nodepair2term_map.binblock_csr_rowptr.Ptr(),
		m_nodepair2term_map.binblock_csr_colptr,
		x.Size());

	// Apply it
	GArray<float> jtj_x;
	jtj_x.create(x.Size());
	spmv_handler.ApplySpMV(x, GArraySlice<float>(jtj_x));

	// Check the error
#if defined(CUDA_DEBUG_SYNC_CHECK)
	cudaSafeCall(cudaDeviceSynchronize());
	cudaSafeCall(cudaGetLastError());
#endif

	// Compare with ground truth
	std::vector<float> spmv_materialized, spmv_result;
	jtj_x.download(spmv_materialized);
	jtj_x_result.Download(spmv_result);

	// Compare it
	const auto relative_err = maxRelativeError(spmv_materialized, spmv_result, 1e-3f);

	LOG(INFO) << "Check done, the relative error between materalized method and matrix free method is " << relative_err;
}