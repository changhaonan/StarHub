#include <mono_star/common/ConfigParser.h>
#include <mono_star/opt/OptimizationProcessorWarpSolver.h>

star::OptimizationProcessorWarpSolver::OptimizationProcessorWarpSolver()
{
	std::cout << "Initialize OptimizationProcessorWarpSolver..." << std::endl;
	auto &config = ConfigParser::Instance();
	m_num_cam = config.num_cam();

	for (auto cam_idx = 0; cam_idx < m_num_cam; ++cam_idx)
	{
		m_cam2world[cam_idx] = config.extrinsic()[cam_idx];
		m_image_height[cam_idx] = config.downsample_img_rows(cam_idx);
		m_image_width[cam_idx] = config.downsample_img_cols(cam_idx);
		m_intrinsic[cam_idx] = config.rgb_intrinsic_downsample(cam_idx);
	}

	m_warp_solver = std::make_shared<WarpSolver>(m_num_cam, m_image_height, m_image_width, m_intrinsic);
}

star::OptimizationProcessorWarpSolver::~OptimizationProcessorWarpSolver()
{
	m_warp_solver->ReleaseBuffer();
}

void star::OptimizationProcessorWarpSolver::ProcessFrame(
	Measure4Solver &measure4solver,
	Render4Solver &render4solver,
	Geometry4Solver &geometry4solver,
	NodeGraph4Solver &node_graph4solver,
	NodeFlow4Solver &nodeflow4solver,
	OpticalFlow4Solver &opticalflow4solver,
	KeyPoint4Solver &keypoint4solver,
	const unsigned frame_idx,
	cudaStream_t stream)
{
	// Collect input
	printf("Optimizer Processing.\n");
	if (frame_idx == 0)
		return; // Operation start from frame 1
	// Set input
	m_warp_solver->SetSolverInputs(
		measure4solver,
		render4solver,
		geometry4solver,
		node_graph4solver,
		nodeflow4solver,
		opticalflow4solver,
		keypoint4solver,
		m_cam2world);

	// Seperate test
	bool opt_success = m_warp_solver->SolveStreamed();
	if (!opt_success)
		std::cout << "Optimization failed. Use previous result instead." << std::endl;

	cudaSafeCall(cudaStreamSynchronize(stream));
}