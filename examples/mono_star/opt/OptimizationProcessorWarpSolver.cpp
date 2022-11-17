#include "star/common/ConfigParser.h"
#include "star/warp_solver/OptimizationProcessorWarpSolver.h"

star::OptimizationProcessorWarpSolver::OptimizationProcessorWarpSolver() {
	m_warp_solver = std::make_shared<WarpSolver>();

	auto& config = ConfigParser::Instance();
	m_num_cam = config.num_cam();

	for (auto cam_idx = 0; cam_idx < m_num_cam; ++cam_idx) {
		m_cam2world[cam_idx] = config.extrinsic()[cam_idx];
	}
}

star::OptimizationProcessorWarpSolver::~OptimizationProcessorWarpSolver() {
	m_warp_solver->ReleaseBuffer();
}

void star::OptimizationProcessorWarpSolver::Process(
	StarStageBuffer& star_stage_buffer_this,
	const StarStageBuffer& star_stage_buffer_prev,
	cudaStream_t stream,
	const unsigned frame_idx
) {

	// Collect input
	printf("Optimizer Processing.\n");
	if (frame_idx == 0) return;  // Operation start from frame 1
	// Set input
	m_warp_solver->SetSolverInputs(
		star_stage_buffer_this.GetMeasureBufferReadOnly()->GenerateMeasure4Solver(m_num_cam),
		star_stage_buffer_prev.GetDynamicGeometryBufferReadOnly()->GenerateRender4Solver(m_num_cam),
		star_stage_buffer_prev.GetDynamicGeometryBufferReadOnly()->GenerateGeometry4Solver(),
		star_stage_buffer_prev.GetDynamicGeometryBufferReadOnly()->GenerateNodeGraph4Solver(),
		star_stage_buffer_this.GetNodeFlowBufferReadOnly()->GenerateNodeFlow4Solver(),
		star_stage_buffer_this.GetOpticalFlowBufferReadOnly()->GenerateOpticalFlow4Solver(m_num_cam),
		m_cam2world
	);

	// Seperate test
	m_warp_solver->SolveStreamed();
	
    // Write to Optimization buffer
	unsigned num_node = star_stage_buffer_prev.GetDynamicGeometryBufferReadOnly()->GenerateNodeGraph4Solver().num_node;
	cudaSafeCall(cudaMemcpyAsync(
		star_stage_buffer_this.GetOptimizationBuffer()->SolvedNodeSE3().Ptr(),
		m_warp_solver->SolvedNodeSE3().Ptr(),
		sizeof(DualQuaternion) * num_node,
		cudaMemcpyDeviceToDevice,
		stream));
	star_stage_buffer_this.GetOptimizationBuffer()->Resize(num_node);

	// #Debug: write potential pixel pair
	for (auto cam_idx = 0; cam_idx < m_num_cam; ++cam_idx) {
		cudaSafeCall(cudaMemcpyAsync(
			star_stage_buffer_this.GetOptimizationBuffer()->PontentialPixelPair(cam_idx).Ptr(),
			m_warp_solver->PotentialPixelPair(cam_idx).Ptr(),
			m_warp_solver->PotentialPixelPair(cam_idx).ByteSize(),
			cudaMemcpyDeviceToDevice,
			stream
		));
		star_stage_buffer_this.GetOptimizationBuffer()->ResizePotentialPixelPair(
			cam_idx, m_warp_solver->PotentialPixelPair(cam_idx).Size());
	}
	

	cudaSafeCall(cudaStreamSynchronize(stream));

	//Check the result
	//Check DQ print
	//#Debug
	//std::vector<DualQuaternion> h_solved_dq;
	//star_stage_buffer_this.GetOptimizationBuffer()->SolvedNodeSE3().View().Download(h_solved_dq);
	//std::cout << "Dq size is " << h_solved_dq.size() << std::endl;
	//std::cout << "Num node is " << num_node << std::endl;
	//for (auto dq : h_solved_dq) {
	//	std::cout << dq.q0.q0.x << "," << dq.q0.q0.y << "," << dq.q0.q0.y << "," << dq.q0.q0.z << "," << dq.q0.q0.w << std::endl;
	//	//std::cout << dq.q0.q0 << ", " << dq.q1.q0 << std::endl;
	//}
}