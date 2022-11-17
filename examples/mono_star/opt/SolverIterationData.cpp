#include "star/warp_solver/SolverIterationData.h"

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