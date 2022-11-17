#include <mono_star/opt/NodeMotionHandler.h>
#include <mono_star/common/Constants.h>
#include <star/math/DualQuaternion.hpp>
#include <device_launch_parameters.h>

namespace star::device
{

	__global__ void ComputeNodeMotionKernel(
		const DualQuaternion *__restrict__ node_se3,
		float3 *__restrict__ T_translation,
		const unsigned node_size)
	{
		const auto idx = threadIdx.x + blockDim.x * blockIdx.x;
		if (idx >= node_size)
			return;
		auto node_dq = node_se3[idx];
		T_translation[idx] = node_dq.se3_matrix().trans;
	}

}

star::NodeMotionHandler::NodeMotionHandler()
{
	m_T_translation.AllocateBuffer(Constants::kMaxNumNodes);
}

star::NodeMotionHandler::~NodeMotionHandler()
{
	m_T_translation.ReleaseBuffer();
}

void star::NodeMotionHandler::SetInputs(
	const NodeGraph4Solver &node_graph4solver,
	const NodeFlow4Solver &nodeflow4solver)
{
	m_node_motion_pred = nodeflow4solver.node_motion_pred;

	m_node_knn = node_graph4solver.nodel_knn;
	m_node_knn_connect_weight = node_graph4solver.node_knn_connect_weight;
	m_node_knn_spatial_weight = node_graph4solver.node_knn_spatial_weight;
}

void star::NodeMotionHandler::UpdateInputs(
	const GArrayView<DualQuaternion> &node_se3)
{
	m_node_se3 = node_se3;
}

star::NodeTranslationTerm2Jacobian star::NodeMotionHandler::Term2JacobianMap() const
{
	NodeTranslationTerm2Jacobian node_translation_term2jacobian;
	node_translation_term2jacobian.node_motion_pred = m_node_motion_pred;
	node_translation_term2jacobian.T_translation = m_T_translation.View();
	return node_translation_term2jacobian;
}

void star::NodeMotionHandler::BuildTerm2Jacobian(cudaStream_t stream)
{
	// Resize
	m_T_translation.ResizeArrayOrException(m_node_se3.Size());

	dim3 blk(128);
	dim3 grid(divUp(m_node_se3.Size(), blk.x));
	device::ComputeNodeMotionKernel<<<grid, blk, 0, stream>>>(
		m_node_se3.Ptr(),
		m_T_translation.Ptr(),
		m_node_se3.Size());
	// Sync and check error
#if defined(CUDA_DEBUG_SYNC_CHECK)
	cudaSafeCall(cudaStreamSynchronize(stream));
	cudaSafeCall(cudaGetLastError());
#endif

	// [Debug]
	residualCheck();
	jacobainTermCheck();
}

void star::NodeMotionHandler::residualCheck()
{
	std::vector<float3> h_T_translation;
	m_T_translation.View().Download(h_T_translation);

	std::vector<float4> h_node_motion_pred;
	m_node_motion_pred.Download(h_node_motion_pred);

	float sum_of_residual = 0.f;
	for (auto i = 0; i < h_node_motion_pred.size(); ++i)
	{
		auto node_motion_pred = h_node_motion_pred[i];
		auto T_translation = h_T_translation[i];
		sum_of_residual += (node_motion_pred.w * node_motion_pred.w * (node_motion_pred.x - T_translation.x) * (node_motion_pred.x - T_translation.x));
		sum_of_residual += (node_motion_pred.w * node_motion_pred.w * (node_motion_pred.y - T_translation.y) * (node_motion_pred.y - T_translation.y));
		sum_of_residual += (node_motion_pred.w * node_motion_pred.w * (node_motion_pred.z - T_translation.z) * (node_motion_pred.z - T_translation.z));
	}

	std::cout << "SOR [NodeTranslation]: " << sum_of_residual << std::endl;
}

void star::NodeMotionHandler::jacobainTermCheck()
{
	STAR_CHECK_EQ(m_node_motion_pred.Size(), m_T_translation.ArraySize());
}