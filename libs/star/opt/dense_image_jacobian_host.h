#pragma once
#include <star/opt/solver_types.h>
#include <star/opt/solver_encode.h>

namespace star
{
	/** Diagonal Jacobian in Debug
	 */
	void updateDenseImageJtJDiagonalHost(
		std::vector<float> &jtj_flatten,
		DenseImageTerm2Jacobian &term2jacobian,
		const floatX<d_dense_image_residual_dim> &weight_square_vec,
		const unsigned inspect_index = 0,
		const bool verbose = false);

	/** Non-Diagonal Jacoba in Debug
	 */
	void updateDenseImageJtJBlockHost(
		const GArrayView<unsigned> &encoded_nodepair,
		std::vector<float> &jtj_flatten,
		const DenseImageTerm2Jacobian &term2jacobian,
		const floatX<d_dense_image_residual_dim> &term_weight_square_vec);

	/** JtResidual
	 */
	void updateDenseImageJtResidualHost(
		std::vector<float> &jt_residual,
		const DenseImageTerm2Jacobian &term2jacobian,
		const floatX<d_dense_image_residual_dim> &term_weight_square_vec);
}