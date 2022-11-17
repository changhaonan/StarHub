#include "star/warp_solver/NodeGraphHandler.h"
#include "star/common/Constants.h"

#include <device_launch_parameters.h>

namespace star {
	namespace device {

		__global__ void forwardWarpSmootherNodeKernel(
			GArrayView<ushort3> node_graph,
			const float4* __restrict__ reference_node_coords,
			const floatX<d_node_knn_size>* __restrict__ node_knn_connect_weight,
			const DualQuaternion* __restrict__ node_se3,
			float3* __restrict__ Ti_xj_array,
			float3* __restrict__ Tj_xj_array,
			unsigned char* __restrict__ validity_indicator_array,
			float* __restrict__ connect_weight,
			const float node_radius_square
		) {
			const auto idx = threadIdx.x + blockIdx.x * blockDim.x;
			if (idx < node_graph.Size()) {
				const ushort3 node_ij_k = node_graph[idx];
				const auto xi = reference_node_coords[node_ij_k.x];
				const auto xj = reference_node_coords[node_ij_k.y];
				DualQuaternion dq_i = node_se3[node_ij_k.x];
				DualQuaternion dq_j = node_se3[node_ij_k.y];
				const mat34 Ti = dq_i.se3_matrix();
				const mat34 Tj = dq_j.se3_matrix();

				const auto Ti_xj = Ti.rot * xj + Ti.trans;
				const auto Tj_xj = Tj.rot * xj + Tj.trans;
				unsigned char validity_indicator = 1;
#define CLIP_FARAWAY_NODEGRAPH_PAIR
#if defined(CLIP_FARAWAY_NODEGRAPH_PAIR)
				if (squared_norm_xyz(xi - xj) > 64 * node_radius_square) {
					validity_indicator = 0;
				}
#endif
				// Save all the data
				Ti_xj_array[idx] = Ti_xj;
				Tj_xj_array[idx] = Tj_xj;
				validity_indicator_array[idx] = validity_indicator;
				connect_weight[idx] = node_knn_connect_weight[node_ij_k.x][node_ij_k.z];
			}
		}

	} // device
} // star


star::NodeGraphHandler::NodeGraphHandler() {
	const auto num_smooth_terms = Constants::kMaxNumNodes * Constants::kNumNodeGraphNeigbours;
	Ti_xj_.AllocateBuffer(num_smooth_terms);
	Tj_xj_.AllocateBuffer(num_smooth_terms);
	m_pair_validity_indicator.AllocateBuffer(num_smooth_terms);
	m_pair_connect_weight.AllocateBuffer(num_smooth_terms);
}

star::NodeGraphHandler::~NodeGraphHandler() {
	Ti_xj_.ReleaseBuffer();
	Tj_xj_.ReleaseBuffer();
	m_pair_validity_indicator.ReleaseBuffer();
	m_pair_connect_weight.ReleaseBuffer();
}

void star::NodeGraphHandler::SetInputs(
	const NodeGraph4Solver& node_graph4solver
) {
	m_node_graph = node_graph4solver.node_graph;
	m_reference_node_coords = node_graph4solver.reference_node_coords;
	m_node_knn_connect_weight = node_graph4solver.node_knn_connect_weight;
	m_node_radius_square = node_graph4solver.node_radius_square;
}

void star::NodeGraphHandler::UpdateInputs(
	const GArrayView<DualQuaternion>& node_se3
) {
	m_node_se3 = node_se3;
}

/* The method to build the term2jacobian
 */
void star::NodeGraphHandler::forwardWarpSmootherNodes(cudaStream_t stream) {
	Ti_xj_.ResizeArrayOrException(m_node_graph.Size());
	Tj_xj_.ResizeArrayOrException(m_node_graph.Size());
	m_pair_validity_indicator.ResizeArrayOrException(m_node_graph.Size());
	m_pair_connect_weight.ResizeArrayOrException(m_node_graph.Size());

	dim3 blk(128);
	dim3 grid(divUp(m_node_graph.Size(), blk.x));
	device::forwardWarpSmootherNodeKernel<<<grid, blk, 0, stream>>>(
		m_node_graph,
		m_reference_node_coords.Ptr(),
		m_node_knn_connect_weight.Ptr(),
		m_node_se3.Ptr(),
		Ti_xj_.Ptr(), Tj_xj_.Ptr(),
		m_pair_validity_indicator.Ptr(),
		m_pair_connect_weight.Ptr(),
		m_node_radius_square
	);

	//Sync and check error
#if defined(CUDA_DEBUG_SYNC_CHECK)
	cudaSafeCall(cudaStreamSynchronize(stream));
	cudaSafeCall(cudaGetLastError());
#endif
}

void star::NodeGraphHandler::BuildTerm2Jacobian(cudaStream_t stream) {
	forwardWarpSmootherNodes(stream);
	jacobianTermCheck();
}

star::NodeGraphRegTerm2Jacobian star::NodeGraphHandler::Term2JacobianMap() const
{
	NodeGraphRegTerm2Jacobian map;
	map.node_graph = m_node_graph;
	map.Ti_xj = Ti_xj_.View();
	map.Tj_xj = Tj_xj_.View();
	map.validity_indicator = m_pair_validity_indicator.View();
	map.connect_weight = m_pair_connect_weight.View();
	return map;
}

void star::NodeGraphHandler::jacobianTermCheck() const {
	auto term2jacobian = Term2JacobianMap();
	// Sanity check
	STAR_CHECK_EQ(term2jacobian.node_graph.Size(), term2jacobian.Ti_xj.Size());
	STAR_CHECK_EQ(term2jacobian.node_graph.Size(), term2jacobian.Tj_xj.Size());
	STAR_CHECK_EQ(term2jacobian.node_graph.Size(), term2jacobian.validity_indicator.Size());
	STAR_CHECK_EQ(term2jacobian.node_graph.Size(), term2jacobian.connect_weight.Size());

	// Check value
	std::vector<float3> h_Ti_xj;
	std::vector<float3> h_Tj_xj;
	term2jacobian.Ti_xj.Download(h_Ti_xj);
	term2jacobian.Tj_xj.Download(h_Tj_xj);
	//for (auto i = 0; i < term2jacobian.Ti_xj.Size(); ++i) {
	//	std::cout << h_Ti_xj[i].x << ", " << h_Ti_xj[i].y << ", " << h_Ti_xj[i].z << ", ";
	//	std::cout << h_Tj_xj[i].x << ", " << h_Tj_xj[i].y << ", " << h_Tj_xj[i].z << std::endl;
	//}
}