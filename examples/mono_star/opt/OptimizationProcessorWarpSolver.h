#pragma once
#include <star/opt/solver_types.h>
#include <mono_star/opt/WarpSolver.h>

namespace star
{
	/* Optimization for WarpField using depth, opticalflow, nodeflow, feature and etc
	 * Apart from WarpField, we can also solve for rigid motion and etc.
	 */
	class OptimizationProcessorWarpSolver
	{
	public:
		using Ptr = std::shared_ptr<OptimizationProcessorWarpSolver>;
		STAR_NO_COPY_ASSIGN_MOVE(OptimizationProcessorWarpSolver);
		OptimizationProcessorWarpSolver();
		~OptimizationProcessorWarpSolver();

		void ProcessFrame(
			Measure4Solver &measure4solver,
			Render4Solver &render4solver,
			Geometry4Solver &geometry4solver,
			NodeGraph4Solver &node_graph4solver,
			NodeFlow4Solver &nodeflow4solver,
			OpticalFlow4Solver &opticalflow4solver,
			const unsigned frame_idx,
			cudaStream_t stream);

	private:
		unsigned m_num_cam;
		Extrinsic m_cam2world[d_max_cam];

		WarpSolver::Ptr m_warp_solver;
	};
}