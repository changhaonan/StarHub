/**
 * @author Wei, Haonan Chang
 * @email chnme40cs@gmail.com
 * @create date 2022-05-04
 * @modify date 2022-05-04
 * @brief NodeGraph regularization term
 */
#pragma once
#include "common/macro_utils.h"
#include "common/ArrayView.h"
#include "common/GBufferArray.h"
#include "math/DualQuaternion.hpp"
#include "star/types/solver_types.h"

namespace star {

	class NodeGraphHandler {
	private:
		// The input data from solver
		GArrayView<DualQuaternion> m_node_se3;
		GArrayView<ushort3> m_node_graph;
		GArrayView<float4> m_reference_node_coords;
		GArrayView<floatX<d_node_knn_size>> m_node_knn_connect_weight;

	public:
		using Ptr = std::shared_ptr<NodeGraphHandler>;
		NodeGraphHandler();
		~NodeGraphHandler();
		STAR_NO_COPY_ASSIGN_MOVE(NodeGraphHandler);

		// The input interface from solver
		void SetInputs(
			const NodeGraph4Solver& node_graph4solver
		);
		void UpdateInputs(
			const GArrayView<DualQuaternion>& node_se3
		);

		// Do a forward warp on nodes
	private:
		GBufferArray<float3> Ti_xj_;
		GBufferArray<float3> Tj_xj_;
		GBufferArray<unsigned char> m_pair_validity_indicator;
		GBufferArray<float> m_pair_connect_weight;
		float m_node_radius_square;

		void forwardWarpSmootherNodes(cudaStream_t stream);
	public:
		void BuildTerm2Jacobian(cudaStream_t stream);
		NodeGraphRegTerm2Jacobian Term2JacobianMap() const;

		// Debug
	public:
		void jacobianTermCheck() const;
	};
}