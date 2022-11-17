#include <star/opt/solver_types.h>
#include <star/math/vector_ops.hpp>
#include <mono_star/opt/SolverIterationData.h>
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

/* The method for construction/destruction, buffer management
 */
star::SolverIterationData::SolverIterationData() : m_is_global_iteration(false) {
	m_updated_warpfield = IterationInputFrom::WarpFieldInit;
	m_newton_iters = 0;
	allocateBuffer();

	// Initialize init as Id
	InitializedAsIdentity(m_node_se3_init.BufferSize());
	// Use config to update correspondingly
	const auto& config = ConfigParser::Instance();
}

star::SolverIterationData::~SolverIterationData() {
	releaseBuffer();
}

void star::SolverIterationData::allocateBuffer() {
	m_node_se3_init.AllocateBuffer(Constants::kMaxNumNodes);
	m_node_se3_0.AllocateBuffer(Constants::kMaxNumNodes);
	m_node_se3_1.AllocateBuffer(Constants::kMaxNumNodes);
	m_warpfield_update.AllocateBuffer(d_node_variable_dim * Constants::kMaxNumNodes);
}

void star::SolverIterationData::releaseBuffer() {
	m_node_se3_0.ReleaseBuffer();
	m_node_se3_1.ReleaseBuffer();
	m_warpfield_update.ReleaseBuffer();
}

/* The processing interface
 */
void star::SolverIterationData::SetWarpFieldInitialValue(const unsigned num_nodes) {
	m_updated_warpfield = IterationInputFrom::WarpFieldInit;
	m_newton_iters = 0;

	// Resize
	m_node_se3_init.ResizeArrayOrException(num_nodes);
	m_node_se3_0.ResizeArrayOrException(num_nodes);
	m_node_se3_1.ResizeArrayOrException(num_nodes);
	m_warpfield_update.ResizeArrayOrException(d_node_variable_dim * num_nodes);

	// Init the penalty constants
	setElasticPenaltyValue(0, m_penalty_constants);
}

star::GArrayView<star::DualQuaternion> star::SolverIterationData::CurrentNodeSE3Input() const {
	switch (m_updated_warpfield) {
	case IterationInputFrom::WarpFieldInit:
		return m_node_se3_init.View();
	case IterationInputFrom::Buffer_0:
		return m_node_se3_0.View();
	case IterationInputFrom::Buffer_1:
		return m_node_se3_1.View();
	default:
		LOG(FATAL) << "Should never happen";
	}
}

void star::SolverIterationData::SanityCheck() const {
	const auto num_nodes = m_node_se3_init.ArraySize();
	STAR_CHECK_EQ(num_nodes, m_node_se3_0.ArraySize());
	STAR_CHECK_EQ(num_nodes, m_node_se3_1.ArraySize());
	STAR_CHECK_EQ(num_nodes * d_node_variable_dim, m_warpfield_update.ArraySize());
}

void star::SolverIterationData::updateIterationFlags() {
	// Update the flag
	if (m_updated_warpfield == IterationInputFrom::Buffer_0) {
		m_updated_warpfield = IterationInputFrom::Buffer_1;
	}
	else {
		// Either init or from buffer 1
		m_updated_warpfield = IterationInputFrom::Buffer_0;
	}

	// Update the iteration counter
	m_newton_iters++;

	// The penalty for next iteration
	setElasticPenaltyValue(m_newton_iters, m_penalty_constants);
}

void star::SolverIterationData::setElasticPenaltyValue(
	int newton_iter,
	PenaltyConstants& constants
) {
	if (!Constants::kUseElasticPenalty) {
		constants.setDefaultValue();
		return;
	}

	if (newton_iter < Constants::kNumGlobalSolverItarations) {
		constants.setGlobalIterationValue();
	}
	else {
		constants.setLocalIterationValue();
	}
}

star::GArraySlice<float> star::SolverIterationData::CurrentWarpFieldUpdateBuffer() {
	return m_warpfield_update.Slice();
}

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