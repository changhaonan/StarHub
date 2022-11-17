#pragma once
#include <star/common/macro_utils.h>
#include <star/common/ArrayView.h>
#include <star/common/GBufferArray.h>
#include <star/opt/solver_types.h>
#include <star/math/DualQuaternion.hpp>

namespace star
{
	/** NodeTranslation Constraint
	 * Support translation only & 6D
	 */
	class NodeMotionHandler
	{
	private:
		// Input from solver
		GArrayView<DualQuaternion> m_node_se3;
		GArrayView<float4> m_node_motion_pred;
		GArrayView<ushortX<d_surfel_knn_size>> m_node_knn;
		GArrayView<floatX<d_surfel_knn_size>> m_node_knn_connect_weight;
		GArrayView<floatX<d_surfel_knn_size>> m_node_knn_spatial_weight;

	public:
		using Ptr = std::shared_ptr<NodeMotionHandler>;
		NodeMotionHandler();
		~NodeMotionHandler();
		STAR_NO_COPY_ASSIGN_MOVE(NodeMotionHandler);

		// The input interface from solver
		void SetInputs(
			const NodeGraph4Solver &node_graph4solver,
			const NodeFlow4Solver &nodeflow4solver);
		void UpdateInputs(
			const GArrayView<DualQuaternion> &node_se3);

	public:
		void BuildTerm2Jacobian(cudaStream_t stream);
		NodeTranslationTerm2Jacobian Term2JacobianMap() const;
		// Debug
	public:
		void residualCheck();
		void jacobainTermCheck();

	private:
		GBufferArray<float3> m_T_translation;
	};
}