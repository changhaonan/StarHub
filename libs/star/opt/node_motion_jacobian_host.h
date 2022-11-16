#pragma once
#include <star/opt/solver_types.h>
#include <star/opt/solver_encode.h>

namespace star {

	/** Diagonal Jacobian
	*/
	void updateNodeTranslationJtJDiagonalHost(
		std::vector<float>& jtj_flatten,
		const NodeTranslationTerm2Jacobian& node_translation_term2jacobian,
		const float weight_square);

	/** JtResidual
	*/
	void updateNodeTranslationJtResidualHost(
		std::vector<float>& jt_residual,
		const NodeTranslationTerm2Jacobian& node_translation_term2jacobian,
		const float weight_square);

}