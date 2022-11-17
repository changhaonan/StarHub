#include "star/warp_solver/SolverIterationData.h"
#include "star/types/solver_types.h"
#include "math/vector_ops.hpp"
#include <device_launch_parameters.h>

namespace star { namespace device {

	__global__ void InitalizeAsIndentityKernel(
		DualQuaternion* __restrict__ node_se3,
		const unsigned num_nodes
	) {
		const auto tidx = threadIdx.x + blockDim.x * blockIdx.x;
		if (tidx >= num_nodes) return;

		node_se3[tidx].set_identity();
	}

	__global__ void ApplyWarpFieldUpdateKernel(
		const DualQuaternion* __restrict__ node_se3,
		const float* __restrict__ warpfield_update,
		DualQuaternion* __restrict__ updated_node_se3,
		const float twist_coef,
		const unsigned node_size,
		const unsigned patch_size
	) {
		const auto tidx = threadIdx.x + blockDim.x * blockIdx.x;
		if (tidx >= node_size) return;
		// Update se3
		float3 twist_rot;
		twist_rot.x = twist_coef * warpfield_update[d_node_variable_dim * tidx];
		twist_rot.y = twist_coef * warpfield_update[d_node_variable_dim * tidx + 1];
		twist_rot.z = twist_coef * warpfield_update[d_node_variable_dim * tidx + 2];

		float3 twist_trans;
		twist_trans.x = twist_coef * warpfield_update[d_node_variable_dim * tidx + 3];
		twist_trans.y = twist_coef * warpfield_update[d_node_variable_dim * tidx + 4];
		twist_trans.z = twist_coef * warpfield_update[d_node_variable_dim * tidx + 5];

		mat34 SE3;
		if (fabsf_sum(twist_rot) < 1e-4f) {
			SE3.rot = mat33::identity();
		}
		else {
			const float angle = norm(twist_rot);
			const float3 axis = 1.0f / angle * twist_rot;

			float c = cosf(angle);
			float s = sinf(angle);
			float t = 1.0f - c;

			SE3.rot.m00() = t * axis.x * axis.x + c;
			SE3.rot.m01() = t * axis.x * axis.y - axis.z * s;
			SE3.rot.m02() = t * axis.x * axis.z + axis.y * s;

			SE3.rot.m10() = t * axis.x * axis.y + axis.z * s;
			SE3.rot.m11() = t * axis.y * axis.y + c;
			SE3.rot.m12() = t * axis.y * axis.z - axis.x * s;

			SE3.rot.m20() = t * axis.x * axis.z - axis.y * s;
			SE3.rot.m21() = t * axis.y * axis.z + axis.x * s;
			SE3.rot.m22() = t * axis.z * axis.z + c;
		}

		SE3.trans = twist_trans;

		mat34 SE3_prev = node_se3[tidx];
		SE3_prev = SE3 * SE3_prev;
		updated_node_se3[tidx] = SE3_prev;
	}

} // namespace device
} // namespace star


void star::SolverIterationData::ApplyWarpFieldUpdate(cudaStream_t stream, float se3_step) {
	// Determine which node list updated to
	const auto init_dq = CurrentNodeSE3Input();
	GArraySlice<DualQuaternion> updated_dq;

	switch (m_updated_warpfield) {
	case IterationInputFrom::WarpFieldInit:  // Follow Buffer_1 (Usage of switch)
	case IterationInputFrom::Buffer_1:
		updated_dq = m_node_se3_0.Slice();
		break;
	case IterationInputFrom::Buffer_0:
		updated_dq = m_node_se3_1.Slice();
		break;
	}

	// Invoke the kernel
	dim3 blk(64);
	dim3 grid(divUp(NumNodes(), blk.x));
	device::ApplyWarpFieldUpdateKernel<<<grid, blk, 0, stream>>>(
		init_dq.Ptr(),
		m_warpfield_update.Ptr(),
		updated_dq.Ptr(),
		se3_step,
		init_dq.Size(),
		d_node_knn_size);

	// Update the flag
	updateIterationFlags();

	// Sync and check error
#if defined(CUDA_DEBUG_SYNC_CHECK)
	cudaSafeCall(cudaStreamSynchronize(stream));
	cudaSafeCall(cudaGetLastError());
#endif
}

void star::SolverIterationData::InitializedAsIdentity(const unsigned num_nodes, cudaStream_t stream) {
	dim3 blk(64);
	dim3 grid(divUp(num_nodes, blk.x));

	device::InitalizeAsIndentityKernel<<<grid, blk, 0, stream>>>(
		m_node_se3_init.Ptr(), num_nodes);

	m_node_se3_init.ResizeArrayOrException(num_nodes);
	// Sync and check error
#if defined(CUDA_DEBUG_SYNC_CHECK)
	cudaSafeCall(cudaStreamSynchronize(stream));
	cudaSafeCall(cudaGetLastError());
#endif
}