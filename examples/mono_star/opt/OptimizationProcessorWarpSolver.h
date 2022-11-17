// #pragma once
// #include <star/opt/solver_types.h>
// #include <mono_star/common/ThreadProcessor.h>
// #include <mono_star/opt/WarpSolver.h>

// namespace star
// {
// 	/* Optimization for WarpField using depth, opticalflow, nodeflow, feature and etc
// 	 * Apart from WarpField, we can also solve for rigid motion and etc.
// 	 */
// 	class OptimizationProcessorWarpSolver : public ThreadProcessor
// 	{
// 	public:
// 		using Ptr = std::shared_ptr<OptimizationProcessorWarpSolver>;
// 		STAR_NO_COPY_ASSIGN_MOVE(OptimizationProcessorWarpSolver);
// 		OptimizationProcessorWarpSolver();
// 		~OptimizationProcessorWarpSolver();

// 		void Process(
// 			StarStageBuffer &star_stage_buffer_this,
// 			const StarStageBuffer &star_stage_buffer_prev,
// 			cudaStream_t stream,
// 			const unsigned frame_idx) override;

// 	private:
// 		unsigned m_num_cam;
// 		Extrinsic m_cam2world[d_max_cam];

// 		WarpSolver::Ptr m_warp_solver;
// 	};
// }