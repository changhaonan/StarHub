#pragma once
#include <mono_star/common/ConfigParser.h>
#include <star/common/macro_utils.h>
#include <star/common/common_types.h>
#include <star/common/data_transfer.h>
#include <star/pcg_solver/BlockPCG.h>
#include <star/opt/solver_types.h>
#include <mono_star/opt/SolverIterationData.h>
#include <mono_star/opt/ImageTermKNNFetcher.h>
#include <mono_star/opt/Node2TermsIndex.h>
#include <mono_star/opt/NodePair2TermsIndex.h>
#include <mono_star/opt/DenseImageHandler.h>
#include <mono_star/opt/NodeGraphHandler.h>
#include <mono_star/opt/NodeMotionHandler.h>
#include <mono_star/opt/PreconditionerRhsBuilder.h>
#include <mono_star/opt/JtJMaterializer.h>

namespace star
{
	class WarpSolver
	{
	public:
		using Ptr = std::shared_ptr<WarpSolver>;
		WarpSolver();
		~WarpSolver();
		STAR_NO_COPY_ASSIGN_MOVE(WarpSolver);
		void AllocateBuffer();
		void ReleaseBuffer();
		void allocatePCGSolverBuffer();
		void releasePCGSolverBuffer();

		// The maps and arrays accessed by solver
		void SetSolverInputs(
			Measure4Solver measure4solver,
			Render4Solver render4solver,
			Geometry4Solver geometry4solver,
			NodeGraph4Solver node_graph4solver,
			NodeFlow4Solver nodeflow4solver,
			OpticalFlow4Solver opticalflow4solver,
			const Extrinsic *camera2world);

		// Acess
		GArrayView<DualQuaternion> SolvedNodeSE3() const { return m_iteration_data.CurrentNodeSE3Input(); }
		GArrayView<ushort4> PotentialPixelPair(const int cam_idx) const
		{
			return m_image_term_knn_fetcher->GetImageTermPixelAndKNN().pixels[cam_idx];
		}

		// API: Solve the warp field using stream
		void SolveStreamed();

		// KNN & Index builder
		void QueryPixelKNN(cudaStream_t stream);
		void buildSolverIndexStreamed();
		void SetNode2TermIndexInput();
		void BuildNode2TermIndex(cudaStream_t stream);
		void BuildNodePair2TermIndexBlocked(cudaStream_t stream);

		// Jacobian related
		void SetPreconditionerBuilderAndJtJApplierInput();
		void SetJtJMaterializerInput();
		void BuildPreconditioner(cudaStream_t stream);
		void ComputeJtResidualIndexed(cudaStream_t stream);
		void MaterializeJtJNondiagonalBlocks(cudaStream_t stream);

		// PCG-solving
		void UpdatePCGSolverStream(cudaStream_t stream);
		void SolvePCGMaterialized(int pcg_iterations = 10);

		// Stream-related
		void initSolverStream();
		void releaseSolverStream();
		void syncAllSolverStream();

	private:
		// Solver method
		void solverIterationGlobalIterationStreamed();
		void solverIterationLocalIterationStreamed();
		void setTermHandlerInput();
		void computeNode2JacobianSync();

		// Input data
		Measure4Solver m_measure4solver;
		Render4Solver m_render4solver;
		Geometry4Solver m_geometry4solver;
		NodeGraph4Solver m_node_graph4solver;
		NodeFlow4Solver m_nodeflow4solver;
		OpticalFlow4Solver m_opticalflow4solver;
		// Owned data
		unsigned m_num_cam;
		unsigned m_image_width[d_max_cam];
		unsigned m_image_height[d_max_cam];
		SolverIterationData m_iteration_data;

		Extrinsic m_world2cam[d_max_cam];
		// Stream
		cudaStream_t m_solver_stream[4];

		// KNN & Index
		GArray2D<KNNAndWeight<d_surfel_knn_size>> m_knn_map[d_max_cam];
		ImageTermKNNFetcher::Ptr m_image_term_knn_fetcher;
		Node2TermsIndex::Ptr m_node2term_index;
		NodePair2TermsIndex::Ptr m_nodepair2term_index;

		// Handler to different terms
		DenseImageHandler::Ptr m_dense_image_handler;
		NodeGraphHandler::Ptr m_node_graph_handler;
		NodeMotionHandler::Ptr m_node_motion_handler;

		// Jacobian handler
		PreconditionerRhsBuilder::Ptr m_preconditioner_rhs_builder;
		JtJMaterializer::Ptr m_jtj_materializer;

		// PCG-solver
		BlockPCG<d_node_variable_dim>::Ptr m_pcg_solver;
	};
}