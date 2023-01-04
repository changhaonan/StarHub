#include <mono_star/opt/WarpSolver.h>
#include <device_launch_parameters.h>
#include <star/visualization/Visualizer.h>

namespace star::device
{
	__global__ void queryPixelKNNKernel(
		cudaTextureObject_t index_map,
		const ushortX<d_surfel_knn_size> *__restrict__ surfel_knn,
		const floatX<d_surfel_knn_size> *__restrict__ surfel_knn_spatial_weight,
		const floatX<d_surfel_knn_size> *__restrict__ surfel_knn_connect_weight,
		PtrStepSz<KNNAndWeight<d_surfel_knn_size>> knn_map)
	{
		const auto x = threadIdx.x + blockDim.x * blockIdx.x;
		const auto y = threadIdx.y + blockDim.y * blockIdx.y;
		if (x < knn_map.cols && y < knn_map.rows)
		{
			KNNAndWeight<d_surfel_knn_size> knn_weight;
			knn_weight.set_invalid();
			const unsigned index = tex2D<unsigned>(index_map, x, y);
			if (index != 0xFFFFFFFF)
			{
				knn_weight.knn = surfel_knn[index];
				knn_weight.spatial_weight = surfel_knn_spatial_weight[index];
				knn_weight.connect_weight = surfel_knn_connect_weight[index];
			}
			knn_map.ptr(y)[x] = knn_weight;

#ifdef OPT_DEBUG_CHECK
			if (index != 0xFFFFFFFF)
			{
				bool flag_zero_connect = true;
				for (auto i = 0; i < d_surfel_knn_size; ++i)
				{
					if (knn_weight.connect_weight[i] != 0.f)
					{
						flag_zero_connect = false;
					}
				}
				if (flag_zero_connect)
				{
					printf("QueryKNN: Zero connect at surfel %d.\n", index);
				}
			}
#endif // OPT_DEBUG_CHECK
		}
	}
}

star::WarpSolver::WarpSolver() : m_whole_sor_init(0.f), m_whole_sor_final(0.f)
{
	// Initialize from config
	auto &config = ConfigParser::Instance();
	m_num_cam = config.num_cam();
	for (auto cam_idx = 0; cam_idx < m_num_cam; ++cam_idx)
	{
		m_image_width[cam_idx] = config.downsample_img_cols(cam_idx);
		m_image_height[cam_idx] = config.downsample_img_rows(cam_idx);
	}

	m_image_term_knn_fetcher = std::make_shared<ImageTermKNNFetcher>();
	// Handlers
	m_dense_image_handler = std::make_shared<DenseImageHandler>();
	m_node_graph_handler = std::make_shared<NodeGraphHandler>();
	m_node_motion_handler = std::make_shared<NodeMotionHandler>();
	m_keypoint_handler = std::make_shared<KeyPointHandler>();
	// Preconditioner
	m_preconditioner_rhs_builder = std::make_shared<PreconditionerRhsBuilder>();
	m_jtj_materializer = std::make_shared<JtJMaterializer>();
	AllocateBuffer();
	// Stream
	initSolverStream();
}

star::WarpSolver::~WarpSolver()
{
	ReleaseBuffer();
	// Stream
	releaseSolverStream();
}

void star::WarpSolver::AllocateBuffer()
{
	// Iterate data
	for (auto cam_idx = 0; cam_idx < m_num_cam; ++cam_idx)
	{
		m_knn_map[cam_idx].create(m_image_height[cam_idx], m_image_width[cam_idx]);
	}

	// Index-related
	m_node2term_index = std::make_shared<Node2TermsIndex>();
	m_node2term_index->AllocateBuffer();
	m_nodepair2term_index = std::make_shared<NodePair2TermsIndex>();
	m_nodepair2term_index->AllocateBuffer();

	// Handler
	m_dense_image_handler->AllocateBuffer();
	// Preconditioner
	m_preconditioner_rhs_builder->AllocateBuffer();
	m_jtj_materializer->AllocateBuffer();
	// PCG
	allocatePCGSolverBuffer();
}

void star::WarpSolver::ReleaseBuffer()
{
	for (auto cam_idx = 0; cam_idx < m_num_cam; ++cam_idx)
	{
		m_knn_map[cam_idx].release();
	}

	// Index-related
	m_node2term_index->ReleaseBuffer();
	m_nodepair2term_index->ReleaseBuffer();

	// Handler
	m_dense_image_handler->ReleaseBuffer();
	// Preconditioner
	m_preconditioner_rhs_builder->ReleaseBuffer();
	m_jtj_materializer->ReleaseBuffer();
	// PCG
	releasePCGSolverBuffer();
}

void star::WarpSolver::SetSolverInputs(
	Measure4Solver measure4solver,
	Render4Solver render4solver,
	Geometry4Solver geometry4solver,
	NodeGraph4Solver node_graph4solver,
	NodeFlow4Solver nodeflow4solver,
	OpticalFlow4Solver opticalflow4solver,
	KeyPoint4Solver keypoint4solver,
	const Extrinsic *camera2world)
{
	m_measure4solver = measure4solver;
	m_render4solver = render4solver;
	m_geometry4solver = geometry4solver;
	m_node_graph4solver = node_graph4solver;
	m_nodeflow4solver = nodeflow4solver;
	m_opticalflow4solver = opticalflow4solver;
	m_keypoint4solver = keypoint4solver;

	// Reinteperate
	GArrayView<float> node_knn_patch_connect_weight =
		ArrayViewCast<floatX<d_node_knn_size>, float>(m_node_graph4solver.node_knn_connect_weight);

	// The iteration data
	m_iteration_data.SetWarpFieldInitialValue(m_node_graph4solver.num_node);

	// Update of extrinsic follows the change o
	for (auto cam_idx = 0; cam_idx < measure4solver.num_cam; ++cam_idx)
	{
		m_world2cam[cam_idx] = camera2world[cam_idx].inverse();
	}
}

void star::WarpSolver::initSolverStream()
{
	// Create the stream
	cudaSafeCall(cudaStreamCreate(&m_solver_stream[0]));
	cudaSafeCall(cudaStreamCreate(&m_solver_stream[1]));
	cudaSafeCall(cudaStreamCreate(&m_solver_stream[2]));
	cudaSafeCall(cudaStreamCreate(&m_solver_stream[3]));

	// Hand in the stream to pcg solver
	// UpdatePCGSolverStream(m_solver_stream[0]);
}

void star::WarpSolver::releaseSolverStream()
{
	// Update 0 stream to pcg solver
	// UpdatePCGSolverStream(0);

	cudaSafeCall(cudaStreamDestroy(m_solver_stream[0]));
	cudaSafeCall(cudaStreamDestroy(m_solver_stream[1]));
	cudaSafeCall(cudaStreamDestroy(m_solver_stream[2]));
	cudaSafeCall(cudaStreamDestroy(m_solver_stream[3]));

	// Assign to null stream
	m_solver_stream[0] = 0;
	m_solver_stream[1] = 0;
	m_solver_stream[2] = 0;
	m_solver_stream[3] = 0;
}

void star::WarpSolver::syncAllSolverStream()
{
	cudaSafeCall(cudaStreamSynchronize(m_solver_stream[0]));
	cudaSafeCall(cudaStreamSynchronize(m_solver_stream[1]));
	cudaSafeCall(cudaStreamSynchronize(m_solver_stream[2]));
	cudaSafeCall(cudaStreamSynchronize(m_solver_stream[3]));
}

void star::WarpSolver::buildSolverIndexStreamed()
{
	// Query knn first
	QueryPixelKNN(m_solver_stream[0]);

	// Dense image index
	for (auto cam_idx = 0; cam_idx < m_num_cam; ++cam_idx) // Per-camera
		m_image_term_knn_fetcher->SetInputs(cam_idx, m_knn_map[cam_idx],
											m_render4solver.index_map[cam_idx], m_opticalflow4solver.opticalflow_map[cam_idx]);
	m_image_term_knn_fetcher->FetchKNNTermSync(m_solver_stream[0]);

	// Build input for handler
	setTermHandlerInput();
	// Build handler knn term
	initTermHandler(m_solver_stream[0]);
	// Node index: has been built in node-graph
	// Build index
	SetNode2TermIndexInput();
	BuildNode2TermIndex(m_solver_stream[0]);
	BuildNodePair2TermIndexBlocked(m_solver_stream[1]);
	cudaSafeCall(cudaStreamSynchronize(m_solver_stream[0]));

	// Sanity Check for Index
#ifdef OPT_DEBUG_CHECK
	m_nodepair2term_index->CheckHalfIndex();
	m_nodepair2term_index->CompactedIndexSanityCheck();
	m_nodepair2term_index->IndexStatistics();
	// m_nodepair2term_index->CompactedIndexLog();
#endif
}

void star::WarpSolver::SetNode2TermIndexInput()
{
	auto sparse_feature_knn_patch = m_keypoint_handler->KNNPatchArray();
	auto image_term_pixel_knn = m_image_term_knn_fetcher->GetImageTermPixelAndKNN();
	unsigned num_nodes = m_node_graph4solver.num_node;
	// Log
	printf("Set: Dense size: %d, node list size: %d, graph size: %d, feature size: %d\n",
		   int(image_term_pixel_knn.surfel_knn_patch_all.Size()),
		   int(num_nodes),
		   int(m_node_graph4solver.node_graph.Size()),
		   int(sparse_feature_knn_patch.Size()));

	m_node2term_index->SetInputs(
		image_term_pixel_knn.surfel_knn_patch_all,
		m_node_graph4solver.node_graph,
		sparse_feature_knn_patch,
		num_nodes);
	m_nodepair2term_index->SetInputs(
		image_term_pixel_knn.surfel_knn_patch_all,
		m_node_graph4solver.node_graph,
		sparse_feature_knn_patch,
		num_nodes);
}

void star::WarpSolver::BuildNode2TermIndex(cudaStream_t stream)
{
	m_node2term_index->BuildIndex(stream);
}

void star::WarpSolver::BuildNodePair2TermIndexBlocked(cudaStream_t stream)
{
	m_nodepair2term_index->BuildHalfIndex(stream);
	m_nodepair2term_index->BuildSymmetricAndRowBlocksIndex(stream);
}

bool star::WarpSolver::SolveStreamed()
{
#ifdef ENABLE_ROBUST_OPT
	m_opt_success = true;
	m_whole_sor_init = 0.f;
	m_whole_sor_final = 0.f;
#endif

	// 1 - Sync before solve
	syncAllSolverStream();
	// 2 - Bulid index
	buildSolverIndexStreamed();

	// 3 - Solve
	for (auto i = 0; i < Constants::kNumGaussNewtonIterations; i++)
	{
		std::cout << "--- Opt Iter " << i << ": ---" << std::endl;
		if (m_iteration_data.IsGlobalIteration())
			solverIterationGlobalIterationStreamed();
		else
			solverIterationLocalIterationStreamed();
	}

	// 4- Sync again before leave
	syncAllSolverStream();

#ifdef ENABLE_ROBUST_OPT
	m_opt_success = m_whole_sor_init >= m_whole_sor_final;
	std::cout << "SOR init: " << m_whole_sor_init << ", SOR final: " << m_whole_sor_final << std::endl;
	return m_opt_success; // If the overall sor decreases
#else
	return true;
#endif
}

void star::WarpSolver::solverIterationGlobalIterationStreamed()
{
	// 1 - Compute term2Jacobian, twist, and etc.
	computeNode2JacobianSync();

	// 2 - Feed input to pre-conditioner & jtj materizer
	SetPreconditionerBuilderAndJtJApplierInput();
	SetJtJMaterializerInput();

	// 3 - Compute JTJ & JTR
	BuildPreconditioner(m_solver_stream[0]);
	ComputeJtResidualIndexed(m_solver_stream[1]);
	MaterializeJtJNondiagonalBlocks(m_solver_stream[2]);
	// Sync
	cudaSafeCall(cudaStreamSynchronize(m_solver_stream[0]));
	cudaSafeCall(cudaStreamSynchronize(m_solver_stream[1]));
	cudaSafeCall(cudaStreamSynchronize(m_solver_stream[2]));

	const auto diagonal_blks = m_preconditioner_rhs_builder->JtJDiagonalBlocks();
	m_jtj_materializer->AssembleBinBlockCSR(diagonal_blks, m_solver_stream[0]);

	// 4 - Solve PCG
	SolvePCGMaterialized(10);

	// 5 - Update x
	m_iteration_data.ApplyWarpFieldUpdate(m_solver_stream[0]);
	cudaSafeCall(cudaStreamSynchronize(m_solver_stream[0]));

#ifdef ENABLE_ROBUST_OPT
	// 6 - Compute SOR
	float whole_sor = computeWholeSOR();
	if (m_whole_sor_init == 0.f)
		m_whole_sor_init = whole_sor;
	m_whole_sor_final = whole_sor;
#endif
}

void star::WarpSolver::solverIterationLocalIterationStreamed()
{
	// 1 - Compute term2Jacobian, twist, and etc.
	computeNode2JacobianSync();

	// 2 - Feed input to pre-conditioner & jtj materizer
	SetPreconditionerBuilderAndJtJApplierInput();
	SetJtJMaterializerInput();

	// 3 - Compute JTJ & JTR
	BuildPreconditioner(m_solver_stream[0]);
	ComputeJtResidualIndexed(m_solver_stream[1]);
	MaterializeJtJNondiagonalBlocks(m_solver_stream[2]);
	// Sync
	cudaSafeCall(cudaStreamSynchronize(m_solver_stream[0]));
	cudaSafeCall(cudaStreamSynchronize(m_solver_stream[1]));
	cudaSafeCall(cudaStreamSynchronize(m_solver_stream[2]));

	const auto diagonal_blks = m_preconditioner_rhs_builder->JtJDiagonalBlocks();
	m_jtj_materializer->AssembleBinBlockCSR(diagonal_blks, m_solver_stream[0]);

	// 4 - Solve PCG
	SolvePCGMaterialized(10);

	// 5 - Update x
	m_iteration_data.ApplyWarpFieldUpdate(m_solver_stream[0]);
	cudaSafeCall(cudaStreamSynchronize(m_solver_stream[0]));

#ifdef ENABLE_ROBUST_OPT
	// 6 - Compute SOR
	float whole_sor = computeWholeSOR();
	if (m_whole_sor_init == 0.f)
		m_whole_sor_init = whole_sor;
	m_whole_sor_final = whole_sor;
#endif
}

void star::WarpSolver::setTermHandlerInput()
{
	m_dense_image_handler->SetInputs(
		m_image_term_knn_fetcher->GetImageTermPixelAndKNN(),
		m_knn_map,
		m_measure4solver,
		m_render4solver,
		m_opticalflow4solver,
		m_world2cam);
	m_node_graph_handler->SetInputs(
		m_node_graph4solver);
	m_node_motion_handler->SetInputs(
		m_node_graph4solver,
		m_nodeflow4solver);
	m_keypoint_handler->SetInputs(
		m_keypoint4solver);
}

void star::WarpSolver::initTermHandler(cudaStream_t stream)
{
	m_keypoint_handler->InitKNNSync(stream);
}

void star::WarpSolver::computeNode2JacobianSync()
{
	const auto node_se3 = m_iteration_data.CurrentNodeSE3Input();

	m_image_term_knn_fetcher->UpdateKnnTermSync(node_se3, m_solver_stream[0]);
	m_dense_image_handler->UpdateInputs(node_se3, m_image_term_knn_fetcher->GetImageTermPixelAndKNN());
	m_node_graph_handler->UpdateInputs(node_se3);
	m_node_motion_handler->UpdateInputs(node_se3);
	m_keypoint_handler->UpdateInputs(node_se3, m_solver_stream[3]);

	// Compute Jacobian
	m_dense_image_handler->ComputeJacobianTermsFixedIndex(m_solver_stream[0]);
	m_node_graph_handler->BuildTerm2Jacobian(m_solver_stream[1]);
	m_node_motion_handler->BuildTerm2Jacobian(m_solver_stream[2]);
	m_keypoint_handler->BuildTerm2Jacobian(m_solver_stream[3]);

	// Synchronization
	cudaSafeCall(cudaStreamSynchronize(m_solver_stream[0]));
	cudaSafeCall(cudaStreamSynchronize(m_solver_stream[1]));
	cudaSafeCall(cudaStreamSynchronize(m_solver_stream[2]));
	cudaSafeCall(cudaStreamSynchronize(m_solver_stream[3]));

	// Debug: Check Residual
#ifdef OPT_DEBUG_CHECK
	m_dense_image_handler->jacobianTermCheck();
	m_node_graph_handler->jacobianTermCheck();
#endif // OPT_DEBUG_CHECK
}

float star::WarpSolver::computeWholeSOR()
{
	// Compute SOR
	float dense_image_sor = m_dense_image_handler->computeSOR();
	float node_motion_sor = m_node_motion_handler->computeSOR();
	float keypoint_sor = m_keypoint_handler->computeSOR();
	// Currently node motion is not in the optimization, so don't count it
	return dense_image_sor + keypoint_sor;
}

void star::WarpSolver::BuildPreconditioner(cudaStream_t stream)
{
	m_preconditioner_rhs_builder->ComputeDiagonalPreconditioner(stream);
}

void star::WarpSolver::SetPreconditionerBuilderAndJtJApplierInput()
{
	// Term2Jacobian
	const auto node2term = m_node2term_index->GetNode2TermMap();
	const auto dense_image_term2jacobian = m_dense_image_handler->Term2JacobianMap();
	const auto reg_term2jacobian = m_node_graph_handler->Term2JacobianMap();
	const auto node_translation_term2jacobian = m_node_motion_handler->Term2JacobianMap();
	const auto feature_term2jacobian = m_keypoint_handler->Term2JacobianMap();

	// Penalty constants
	const auto penalty_constants = m_iteration_data.CurrentPenaltyConstants();

	// Hand in the input to preconditioner builder
	m_preconditioner_rhs_builder->SetInputs(
		node2term,
		dense_image_term2jacobian,
		reg_term2jacobian,
		node_translation_term2jacobian,
		feature_term2jacobian,
		penalty_constants);
}

void star::WarpSolver::SetJtJMaterializerInput()
{
	// Index: Node2Term & NodePair2Term
	const auto node2term = m_node2term_index->GetNode2TermMap();
	const auto nodepair2term = m_nodepair2term_index->GetNodePair2TermMap();

	// Term2Jacobian
	const auto dense_image_term2jacobian = m_dense_image_handler->Term2JacobianMap();
	const auto reg_term2jacobian = m_node_graph_handler->Term2JacobianMap();
	const auto node_translation_term2jacobian = m_node_motion_handler->Term2JacobianMap();
	const auto feature_term2jacobian = m_keypoint_handler->Term2JacobianMap();

	// Penalty constants
	const auto penalty_constants = m_iteration_data.CurrentPenaltyConstants();

	// Hand in the input to jtj Materializer
	m_jtj_materializer->SetInputs(
		nodepair2term,
		dense_image_term2jacobian,
		reg_term2jacobian,
		node_translation_term2jacobian,
		feature_term2jacobian,
		node2term,
		penalty_constants);
}

void star::WarpSolver::ComputeJtResidualIndexed(cudaStream_t stream)
{
	m_preconditioner_rhs_builder->ComputeJtResidualIndexed(stream);
}

void star::WarpSolver::MaterializeJtJNondiagonalBlocks(cudaStream_t stream)
{
	m_jtj_materializer->BuildMaterializedJtJNondiagonalBlocks(stream);
}

void star::WarpSolver::QueryPixelKNN(cudaStream_t stream)
{
	dim3 blk(16, 16);
	dim3 grid;
	for (auto cam_idx = 0; cam_idx < m_num_cam; ++cam_idx)
	{ // Per-camera
		grid.x = divUp(m_image_width[cam_idx], blk.x);
		grid.y = divUp(m_image_height[cam_idx], blk.y);
		device::queryPixelKNNKernel<<<grid, blk, 0, stream>>>(
			m_render4solver.index_map[cam_idx],
			m_geometry4solver.surfel_knn.Ptr(),
			m_geometry4solver.surfel_knn_spatial_weight.Ptr(),
			m_geometry4solver.surfel_knn_connect_weight.Ptr(),
			m_knn_map[cam_idx]);
	}

	// Sync and check error
#if defined(CUDA_DEBUG_SYNC_CHECK)
	cudaSafeCall(cudaStreamSynchronize(stream));
	cudaSafeCall(cudaGetLastError());
#endif
}

void star::WarpSolver::allocatePCGSolverBuffer()
{
	const auto max_matrix_size = d_node_variable_dim * Constants::kMaxNumNodes;
	m_pcg_solver = std::make_shared<BlockPCG<d_node_variable_dim>>(max_matrix_size);
}

void star::WarpSolver::releasePCGSolverBuffer()
{
}