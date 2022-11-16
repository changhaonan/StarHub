#pragma once
#include <star/opt/solver_types.h>
#include <star/opt/solver_encode.h>
#include <star/opt/node_graph_reg_jacobian.cuh>

namespace star {

	/** Diagonal JtJ
	*/
	void updateRegJtJDiagonalHost(
		std::vector<float>& jtj_flatten,
		const NodeGraphRegTerm2Jacobian& node_graph_reg_term2jacobian,
		const float weight_square,
		const bool verbose = false);

	/** Non-Diagonal JtJ
	*/
	void updateRegJtJBlockHost(
		const GArrayView<unsigned>& encoded_nodepair,
		std::vector<float>& jtj_flatten,
		const NodeGraphRegTerm2Jacobian& node_graph_reg_term2jacobian,
		const float weight_square);

	/** JtResidual
	*/
	void updateRegJtResidualHost(
		std::vector<float>& jt_residual,
		const NodeGraphRegTerm2Jacobian& node_graph_reg_term2jacobian,
		const float weight_square);
}