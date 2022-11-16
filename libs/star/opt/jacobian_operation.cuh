#pragma once
#include <star/common/types/typeX.h>
#include <device_launch_parameters.h>

namespace star { namespace device {

	/** JtJ Related
	*/
	__device__ __forceinline__ void fillScalarJtJToSharedBlock(
		const float* __restrict__ jacobian,
		float* __restrict__ shared_jtj_blks,
		const float weight_square,
		const unsigned jacobian_dim,
		const unsigned warp_size
	) {
		for (int jac_row = 0; jac_row < jacobian_dim; jac_row++) {
			for (int jac_col = 0; jac_col < jacobian_dim; jac_col++) {
				shared_jtj_blks[(jacobian_dim * jac_row + jac_col) * warp_size + threadIdx.x] = weight_square * jacobian[jac_col] * jacobian[jac_row];
			}
		}
	}

	/** \brief: Fill Diagonal JTJ to shared block, warp-level parallism.
	* jacobian is (jacobian_dim, 3), jacobian_channelled is 3 channel jacobian flattened
	*/
	__device__ __forceinline__ void fillThreeChannelledJtJToSharedBlock(
		const float* __restrict__ jacobian_channelled,
		float* __restrict__ shared_jtj_blks,
		const floatX<3> weight_square_vec,
		const unsigned jacobian_dim,
		const unsigned warp_size
	) {
		// The first iteration: assign
		const float* jacobian_0 = jacobian_channelled;
		for (int jac_row = 0; jac_row < jacobian_dim; jac_row++) {
			for (int jac_col = 0; jac_col < jacobian_dim; jac_col++) {
				shared_jtj_blks[(jacobian_dim * jac_row + jac_col) * warp_size + threadIdx.x] = weight_square_vec[0] * jacobian_0[jac_col] * jacobian_0[jac_row];
			}
		}

		// The next 2 iterations: plus
		const float* jacobian_1 = &(jacobian_channelled[1 * jacobian_dim]);
		for (int jac_row = 0; jac_row < jacobian_dim; jac_row++) {
			for (int jac_col = 0; jac_col < jacobian_dim; jac_col++) {
				shared_jtj_blks[(jacobian_dim * jac_row + jac_col) * warp_size + threadIdx.x] += weight_square_vec[1] * jacobian_1[jac_col] * jacobian_1[jac_row];
			}
		}
		const float* jacobian_2 = &(jacobian_channelled[2 * jacobian_dim]);
		for (int jac_row = 0; jac_row < jacobian_dim; jac_row++) {
			for (int jac_col = 0; jac_col < jacobian_dim; jac_col++) {
				shared_jtj_blks[(jacobian_dim * jac_row + jac_col) * warp_size + threadIdx.x] += weight_square_vec[2] * jacobian_2[jac_col] * jacobian_2[jac_row];
			}
		}
	}

	__device__ __forceinline__ void fillThreeChannelledJtJToSharedBlock(
		const float* __restrict__ jacobian_channelled,
		float* __restrict__ shared_jtj_blks,
		const float weight_square,
		const unsigned jacobian_dim,
		const unsigned warp_size
	) {
		// The first iteration: assign
		const float* jacobian_0 = jacobian_channelled;
		for (int jac_row = 0; jac_row < jacobian_dim; jac_row++) {
			for (int jac_col = 0; jac_col < jacobian_dim; jac_col++) {
				shared_jtj_blks[(jacobian_dim * jac_row + jac_col) * warp_size + threadIdx.x] = weight_square * jacobian_0[jac_col] * jacobian_0[jac_row];
			}
		}

		// The next 2 iterations: plus
		const float* jacobian_1 = &(jacobian_channelled[1 * jacobian_dim]);
		for (int jac_row = 0; jac_row < jacobian_dim; jac_row++) {
			for (int jac_col = 0; jac_col < jacobian_dim; jac_col++) {
				shared_jtj_blks[(jacobian_dim * jac_row + jac_col) * warp_size + threadIdx.x] += weight_square * jacobian_1[jac_col] * jacobian_1[jac_row];
			}
		}
		const float* jacobian_2 = &(jacobian_channelled[2 * jacobian_dim]);
		for (int jac_row = 0; jac_row < jacobian_dim; jac_row++) {
			for (int jac_col = 0; jac_col < jacobian_dim; jac_col++) {
				shared_jtj_blks[(jacobian_dim * jac_row + jac_col) * warp_size + threadIdx.x] += weight_square * jacobian_2[jac_col] * jacobian_2[jac_row];
			}
		}
	}

}
}