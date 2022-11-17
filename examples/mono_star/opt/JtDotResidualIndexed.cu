#include <star/common/device_intrinsics.cuh>
#include <star/opt/jacobian_utils.cuh>
#include <mono_star/opt/PreconditionerRhsBuilder.h>
#include <device_launch_parameters.h>

namespace star::device
{

	enum
	{
		warp_size = 32, // Note: Has to be 32 here
		num_warps = 4,
		thread_blk_size = num_warps * warp_size,
		jt_dot_blk_size = d_node_variable_dim
	};

	// Note: filling pattern is highly associated with implementation
	__device__ __forceinline__ void fillScalarJtResidualToSharedBlock(
		const float *__restrict__ jt_redisual,
		float *__restrict__ shared_blks,
		const float weight_square)
	{
#pragma unroll
		for (auto i = 0; i < jt_dot_blk_size; i++)
		{
			shared_blks[i * thread_blk_size + threadIdx.x] = -weight_square * jt_redisual[i];
		}
	}

	__device__ __forceinline__ void fillScalarJtResidualToSharedBlock(
		const float *__restrict__ jt_redisual,
		float *__restrict__ shared_blks)
	{
#pragma unroll
		for (auto i = 0; i < jt_dot_blk_size; i++)
		{
			shared_blks[i * thread_blk_size + threadIdx.x] = -jt_redisual[i];
		}
	}

	__global__ void computeJtResidualWithIndexKernel(
		const Node2TermsIndex::Node2TermMap node2term,
		const Term2JacobianMaps term2jacobian,
		float *__restrict__ jt_residual,
		const PenaltyConstants constants = PenaltyConstants())
	{
		const auto node_idx = blockIdx.x;
		const auto term_begin = node2term.offset[node_idx];
		const auto term_end = node2term.offset[node_idx + 1];
		const auto term_size = term_end - term_begin;
		const auto padded_term_size = thread_blk_size * ((term_size + thread_blk_size - 1) / thread_blk_size);
		const auto warp_id = threadIdx.x / warp_size;
		const auto lane_id = threadIdx.x % warp_size;

		extern __shared__ char shared_array[];
		// The memory for store the JtResidual result of each threads
		float *shared_blks = (float *)shared_array;
		float *shared_warp_tmp = (float *)(shared_array + sizeof(float) * jt_dot_blk_size * thread_blk_size);
		// The memory to perform the reduction
		float *reduced_blks = (float *)(shared_array + sizeof(float) * jt_dot_blk_size * thread_blk_size + sizeof(float) * num_warps);
#pragma unroll
		for (auto iter = threadIdx.x; iter < jt_dot_blk_size; iter += thread_blk_size)
		{
			reduced_blks[iter] = 0.0f;
		}
		__syncthreads();

		// The warp compute terms in the multiple of 32 (the warp size)
		for (auto iter = threadIdx.x; iter < padded_term_size; iter += thread_blk_size)
		{
			// The global term index
			bool term_valid = true;

			// Do computation when the term is inside
			if (iter < term_size)
			{
				// Query the term type
				const auto term_idx = node2term.term_index[term_begin + iter];
				unsigned typed_term_idx;
				TermType term_type;
				query_typed_index(term_idx, node2term.term_offset, term_type, typed_term_idx);

				// Do computation given term_type
				switch (term_type)
				{
				case TermType::DenseImage:
				{
					float term_jt_residual[jt_dot_blk_size] = {0};
					computeDenseImageJacobianTransposeDot(
						term2jacobian.dense_image_term, node_idx, typed_term_idx, term_jt_residual,
						constants.DenseImageSquaredVec());
					fillScalarJtResidualToSharedBlock(term_jt_residual, shared_blks);
				}
				break;
				case TermType::Reg:
				{
					float term_jt_residual[jt_dot_blk_size] = {0};
					computeRegJtResidual(term2jacobian.node_graph_reg_term, node_idx, typed_term_idx, term_jt_residual, jt_dot_blk_size);
					fillScalarJtResidualToSharedBlock(term_jt_residual, shared_blks, constants.RegSquared());
				}
				break;
				case TermType::NodeTranslation:
				{
					float term_jt_residual[jt_dot_blk_size] = {0};
					computeNodeMotionJtResidual(term2jacobian.node_translation_term, node_idx, typed_term_idx, term_jt_residual);
					fillScalarJtResidualToSharedBlock(term_jt_residual, shared_blks, constants.NodeTranslationSquared());
				}
				break;
				case TermType::Feature:
				{
					float term_jt_residual[jt_dot_blk_size] = {0};
					// TODO: to implement
					fillScalarJtResidualToSharedBlock(term_jt_residual, shared_blks, constants.FeatureSquared());
				}
				break;
				default:
					term_valid = false;
					break;
				}
			}

			// Do a reduction to reduced_men
			__syncthreads();
			for (int i = 0; i < jt_dot_blk_size; i++)
			{
				float data = (iter < term_size && term_valid) ? shared_blks[i * thread_blk_size + threadIdx.x] : 0.0f;
				data = warp_scan(data);
				if (lane_id == warp_size - 1)
				{
					shared_warp_tmp[warp_id] = data;
				}

				__syncthreads();
				data = threadIdx.x < num_warps ? shared_warp_tmp[threadIdx.x] : 0.0f;
				data = warp_scan(data);
				if (threadIdx.x == warp_size - 1)
				{
					reduced_blks[i] += data;
				}
				__syncthreads();
			}
		}

		// All the terms that contribute to this value is done, store to global memory
		if (threadIdx.x < jt_dot_blk_size)
		{
			jt_residual[jt_dot_blk_size * node_idx + threadIdx.x] = reduced_blks[threadIdx.x];
			auto jtr_idx = jt_dot_blk_size * node_idx + threadIdx.x;
		}
	}
}

// Compute the Jt.dot(residual) using the index from node to term
void star::PreconditionerRhsBuilder::ComputeJtResidualIndexed(cudaStream_t stream)
{
	const auto num_nodes = m_node2term_map.offset.Size() - 1;
	m_jt_residual.ResizeArrayOrException(num_nodes * device::jt_dot_blk_size);
	dim3 blk(device::thread_blk_size);
	dim3 grid(num_nodes);
	size_t shared_mem_size = sizeof(float) * device::jt_dot_blk_size * (device::thread_blk_size + 1) + sizeof(float) * device::num_warps;
	device::computeJtResidualWithIndexKernel<<<grid, blk, shared_mem_size, stream>>>(
		m_node2term_map,
		m_term2jacobian_map,
		m_jt_residual.Ptr(),
		m_penalty_constants);

	// Debug
#ifdef OPT_DEBUG_CHECK
	jacobianTransposeResidualSanityCheck();
#endif // OPT_DEBUG_CHECK
}

void star::PreconditionerRhsBuilder::ComputeJtResidualLocalIteration(cudaStream_t stream)
{
	ComputeJtResidualIndexed(stream);

	// Sync and check error
#if defined(CUDA_DEBUG_SYNC_CHECK)
	cudaSafeCall(cudaStreamSynchronize(stream));
	cudaSafeCall(cudaGetLastError());
#endif
}