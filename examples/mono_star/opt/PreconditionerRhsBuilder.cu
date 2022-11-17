#include "common/device_intrinsics.cuh"
#include "star/common/global_configs.h"
#include "star/warp_solver/PreconditionerRhsBuilder.h"
#include "star/warp_solver/Node2TermsIndex.h"
#include "star/types/term_offset_types.h"
#include "star/warp_solver/utils/jacobian_utils.cuh"
//#include "star/warp_solver/utils/jacobian_operation.cuh"
#include "star/warp_solver/PenaltyConstants.h"
#include <device_launch_parameters.h>

namespace star { namespace device {

	enum {
		warp_size = opt_warp_size
	};

	/** Fill-in JtJ Related
	*/
	__device__ __forceinline__ void fillScalarJtJToSharedBlock(
		const float* jacobian,
		float* shared_jtj_blks,
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
		const float* jacobian_channelled,
		float* shared_jtj_blks,
		const floatX<3>& weight_square_vec,
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
		const float* jacobian_channelled,
		float* shared_jtj_blks,
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

	__global__ void computeBlockDiagonalPreconditionerKernel(
		const Node2TermsIndex::Node2TermMap node2term,
		const Term2JacobianMaps term2jacobian,
		float* __restrict__ diagonal_preconditioner,
		const PenaltyConstants constants = PenaltyConstants()
	) {
		// 1 - Preparation
		const auto node_idx = blockIdx.x;
		const auto term_begin = node2term.offset[node_idx];
		const auto term_end = node2term.offset[node_idx + 1];
		const auto term_size = term_end - term_begin;
		const auto padded_term_size = warp_size * ((term_size + warp_size - 1) / warp_size);
		extern __shared__ char shared_array[];
		float* shared_blks = (float*)shared_array;
		float* reduced_blks = (float*)(preconditioner_blk_size * warp_size * sizeof(float) + shared_array);

#pragma unroll
		for (auto iter = threadIdx.x; iter < preconditioner_blk_size; iter += warp_size) {
			reduced_blks[iter] = 0.0f;
		}
		for (auto i = 0; i < preconditioner_blk_size; ++i) {
			shared_blks[i * warp_size + threadIdx.x] = 0.f;
		}
		__syncthreads();

		// 2 - The warp compute terms in the multiple of (the warp size)
		for (auto iter = threadIdx.x; iter < padded_term_size; iter += warp_size)
		{
			// The global term index
			bool term_valid = true;

			// 2.1 - Do computation when the term is inside
			if (iter < term_size)
			{
				// 2.1.1 - Query the term type
				const auto term_idx = node2term.term_index[term_begin + iter];
				unsigned typed_term_idx;
				TermType term_type;
				query_typed_index(term_idx, node2term.term_offset, term_type, typed_term_idx);

				// 2.1.2 - Do computation given term_type
				switch (term_type)
				{
				case TermType::DenseImage:
				{
					// 2.1.2.1. DenseImage term
					float jacobian_channelled[d_dense_image_residual_dim * d_node_variable_dim] = { 0 };
					float weight = 0.f;
					computeDenseImageJtJDiagonalJacobian(term2jacobian.dense_image_term, node_idx, typed_term_idx, jacobian_channelled, weight);					
					fillThreeChannelledJtJToSharedBlock(jacobian_channelled, shared_blks, constants.DenseImageSquaredVec(), d_node_variable_dim, warp_size);
					//// [Debug]
					//size_t idx_in_blk = 0;
					//size_t element_idx = node_idx * preconditioner_blk_size + idx_in_blk;
					//size_t idx_in_shared = idx_in_blk * warp_size + threadIdx.x;
					//if (element_idx == 72) {
					//	for (auto channel = 0; channel < d_dense_image_residual_dim; ++channel) {
					//		printf("[%d], GPU: DenseImage Term: %d, jac at node_%d: channel: %d, weight: (%f, %f) is : (%f, %f, %f, %f, %f, %f).\n",
					//			int(element_idx), int(typed_term_idx),
					//			int(node_idx), int(channel),
					//			constants.DenseImageSquaredVec()[channel], weight * weight,
					//			jacobian_channelled[channel * d_node_variable_dim + 0] / weight, 
					//			jacobian_channelled[channel * d_node_variable_dim + 1] / weight, 
					//			jacobian_channelled[channel * d_node_variable_dim + 2] / weight,
					//			jacobian_channelled[channel * d_node_variable_dim + 3] / weight, 
					//			jacobian_channelled[channel * d_node_variable_dim + 4] / weight, 
					//			jacobian_channelled[channel * d_node_variable_dim + 5] / weight);
					//	}
					//}
				}
				break;
				case TermType::Reg:
				{
					// 2.1.2.1. Regulation term
					float jacobian_channelled[3 * d_node_variable_dim] = { 0 };
					computeRegJtJDiagonalJacobian(term2jacobian.node_graph_reg_term, node_idx, typed_term_idx, jacobian_channelled);
					fillThreeChannelledJtJToSharedBlock(jacobian_channelled, shared_blks, constants.RegSquared(), d_node_variable_dim, warp_size);
					//// [Debug]
					////if (node_idx == 0) {
					//	//printf("Reg: GPU: term: %d, val: %f, reg jacobian is: (%f, %f, %f, %f, %f, %f).\n",
					//	//	typed_term_idx, shared_blks[threadIdx.x],
					//	//	jacobian_channelled[0], jacobian_channelled[1], jacobian_channelled[2],
					//	//	jacobian_channelled[3], jacobian_channelled[4], jacobian_channelled[5]
					//	//);
					////}
				}
				break;
				case TermType::NodeTranslation:
				{
					// 2.1.2.1. Node translation term
					float jacobian_channelled[3 * d_node_variable_dim] = { 0 };
					computeNodeTranslationJtJDiagonalJacobian(term2jacobian.node_translation_term, node_idx, typed_term_idx, jacobian_channelled);
					fillThreeChannelledJtJToSharedBlock(jacobian_channelled, shared_blks, constants.NodeTranslationSquared(), d_node_variable_dim, warp_size);
					// [Debug]
					//if (node_idx == 0) {
						//printf("NodeT: GPU: term: %d, val: %f, node jacobian is: (%f, %f, %f, %f, %f, %f).\n",
						//	typed_term_idx, shared_blks[threadIdx.x],
						//	jacobian_channelled[0], jacobian_channelled[1], jacobian_channelled[2],
						//	jacobian_channelled[3], jacobian_channelled[4], jacobian_channelled[5]
						//);
					//}
				}
				break;
				case TermType::Feature:
				{
					float jacobian[d_node_variable_dim] = { 0 };
					// TODO: implement P2P jacobian
					fillScalarJtJToSharedBlock(jacobian, shared_blks, constants.FeatureSquared(), d_node_variable_dim, warp_size);
				}
				break;
				case TermType::Invalid:
					term_valid = false;
					break;
				} // the switch of types
			}
			
			// Do a reduction to reduced_men
			__syncthreads();
			for (int i = 0; i < preconditioner_blk_size; i++) {
				float data = (iter < term_size&& term_valid) ? shared_blks[i * warp_size + threadIdx.x] : 0.0f;
				//if (node_idx == 2 && i == 0) {
				//	printf("idx: %d, val: %f.\n", int(i), float(data));
				//}
				data = warp_scan(data);
				if (threadIdx.x == warp_size - 1) {
					reduced_blks[i] += data;
					//if (node_idx == 2 && i == 0) {
					//	printf("val: %f increased by %f.\n", reduced_blks[0], data);
					//}
				}
				// Another sync here for reduced mem?
				//__syncthreads();
			}
		}
		
		// add small offset to diagonal elements
		for (unsigned i = threadIdx.x; i < d_node_variable_dim; i += warp_size) {
			reduced_blks[i * (d_node_variable_dim + 1)] += 1e-3f;  // 1e-3f or 1.f
		}
		__syncthreads();
		
		// All the terms that contribute to this value is done, store to global memory
#pragma unroll
		for (int i = threadIdx.x; i < preconditioner_blk_size; i += warp_size) {
			diagonal_preconditioner[preconditioner_blk_size * node_idx + i] = reduced_blks[i];
		}
	}

} // namespace device
} // namespace star


void star::PreconditionerRhsBuilder::ComputeDiagonalBlocks(cudaStream_t stream) {
	const auto num_nodes = m_node2term_map.offset.Size() - 1;
	m_block_preconditioner.ResizeArrayOrException(num_nodes * preconditioner_blk_size);
	dim3 blk(device::warp_size);
	dim3 grid(num_nodes);
	size_t shared_mem_size = (device::warp_size + 1) * preconditioner_blk_size * sizeof(float);
	device::computeBlockDiagonalPreconditionerKernel<<<grid, blk, shared_mem_size, stream>>>(
		m_node2term_map,
		m_term2jacobian_map,
		m_block_preconditioner.Ptr(),
		m_penalty_constants
	);

	// Sync and check error
#if defined(CUDA_DEBUG_SYNC_CHECK)
	cudaSafeCall(cudaStreamSynchronize(stream));
	cudaSafeCall(cudaGetLastError());
#endif

	// Sanity check
#ifdef OPT_DEBUG_CHECK
	diagonalPreconditionerSanityCheck();
#endif // OPT_DEBUG_CHECK

}

void star::PreconditionerRhsBuilder::ComputeDiagonalPreconditionerGlobalIteration(cudaStream_t stream) {
	// Check the constants
	//STAR_CHECK(m_penalty_constants.Density() < 1e-7f);

	// Do computation
	//const auto num_nodes = m_node2term_map.offset.Size() - 1;
	//m_block_preconditioner.ResizeArrayOrException(num_nodes * device::preconditioner_blk_size);
	//dim3 blk(device::warp_size);
	//dim3 grid(num_nodes);
	//device::computeBlockDiagonalPreconditionerGlobalIterationKernel<<<grid, blk, 0, stream>>>(
	//	m_node2term_map,
	//	m_term2jacobian_map,
	//	m_block_preconditioner.Ptr(),
	//	m_penalty_constants
	//);

	// Sync and check error
//#if defined(CUDA_DEBUG_SYNC_CHECK)
//	cudaSafeCall(cudaStreamSynchronize(stream));
//	cudaSafeCall(cudaGetLastError());
//#endif
//
//	// Do inversion
//	ComputeDiagonalPreconditionerInverse(stream);
}

void star::PreconditionerRhsBuilder::ComputeDiagonalPreconditionerInverse(cudaStream_t stream) {
	m_preconditioner_inverse_handler->SetInput(m_block_preconditioner.View());
	m_preconditioner_inverse_handler->PerformDiagonalInverse(stream);
}