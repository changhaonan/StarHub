#include <mono_star/opt/WarpSolver.h>

void star::WarpSolver::UpdatePCGSolverStream(cudaStream_t stream) {
	m_pcg_solver->UpdateCudaStream(stream);
}

void star::WarpSolver::SolvePCGMaterialized(int pcg_iterations) {
	// Prepare the data
	const auto inversed_diagonal_preconditioner = m_preconditioner_rhs_builder->InversedPreconditioner();
	const auto rhs = m_preconditioner_rhs_builder->JtDotResidualValue();
	ApplySpMVBase<d_node_variable_dim>::Ptr apply_spmv_handler = m_jtj_materializer->GetSpMVHandler();
	GArraySlice<float> updated_warpfield = m_iteration_data.CurrentWarpFieldUpdateBuffer();

	// sanity check
	STAR_CHECK_EQ(rhs.Size(), apply_spmv_handler->MatrixSize());
	STAR_CHECK_EQ(updated_warpfield.Size(), apply_spmv_handler->MatrixSize());
	STAR_CHECK_EQ(inversed_diagonal_preconditioner.Size(), apply_spmv_handler->MatrixSize() * d_node_variable_dim);

	cudaSafeCall(cudaDeviceSynchronize());
	cudaSafeCall(cudaGetLastError());
	// Check nan
	std::vector<float> h_rhs;
	unsigned count = 0;
	rhs.Download(h_rhs);
	for (auto v : h_rhs) {
		if (std::isnan(v)) {
			printf("Found nan at %d.\n", count);
			exit(-1);
		}
		count++;
	}
	// Hand in to warp solver and solve it
	m_pcg_solver->SetSolverInput(inversed_diagonal_preconditioner, apply_spmv_handler, rhs, updated_warpfield);
	m_pcg_solver->Solve(pcg_iterations);
}