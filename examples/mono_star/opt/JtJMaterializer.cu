#include <device_launch_parameters.h>
#include <star/common/device_intrinsics.cuh>
#include <star/opt/jacobian_utils.cuh>
#include <star/opt/jacobian_operation.cuh>
#include <mono_star/opt/JtJMaterializer.h>

namespace star::device
{
	enum
	{
		jtj_blk_size = d_node_variable_dim_square,
		warp_size = 32,
		half_warp_size = 16
	};

	__device__ __forceinline__ void fillScalarJtJToLocalBlock(
		const float *__restrict__ jacobian,
		float *__restrict__ local_jtj_blks,
		const float weight_square = 1.0f)
	{
		for (int jac_row = 0; jac_row < d_node_variable_dim; jac_row++)
		{
#pragma unroll
			for (int jac_col = 0; jac_col < d_node_variable_dim; jac_col++)
			{
				local_jtj_blks[d_node_variable_dim * jac_row + jac_col] = weight_square * jacobian[jac_col] * jacobian[jac_row];
			}
		}
	}

	__device__ __forceinline__ void fillThreeChannelledJtJToLocalBlock(
		const float *__restrict__ jacobian_channelled,
		float *__restrict__ local_jtj_blks,
		const floatX<3> weight_square_vec)
	{
		// The first iteration: assign
		const float *jacobian_0 = jacobian_channelled;
		for (int jac_row = 0; jac_row < d_node_variable_dim; jac_row++)
		{
			for (int jac_col = 0; jac_col < d_node_variable_dim; jac_col++)
			{
				local_jtj_blks[d_node_variable_dim * jac_row + jac_col] = weight_square_vec[0] * jacobian_0[jac_col] * jacobian_0[jac_row];
			}
		}

		// The next 2 iterations: plus
		const float *jacobian_1 = &(jacobian_channelled[1 * d_node_variable_dim]);
		for (int jac_row = 0; jac_row < d_node_variable_dim; jac_row++)
		{
			for (int jac_col = 0; jac_col < d_node_variable_dim; jac_col++)
			{
				local_jtj_blks[d_node_variable_dim * jac_row + jac_col] += weight_square_vec[1] * jacobian_1[jac_col] * jacobian_1[jac_row];
			}
		}
		const float *jacobian_2 = &(jacobian_channelled[2 * d_node_variable_dim]);
		for (int jac_row = 0; jac_row < d_node_variable_dim; jac_row++)
		{
			for (int jac_col = 0; jac_col < d_node_variable_dim; jac_col++)
			{
				local_jtj_blks[d_node_variable_dim * jac_row + jac_col] += weight_square_vec[2] * jacobian_2[jac_col] * jacobian_2[jac_row];
			}
		}
	}

	// TODO: What is this launch bounds?
	__global__ void
	//__launch_bounds__(32, 32)
	computeJtJNonDiagonalBlockKernel(
		const NodePair2TermsIndex::NodePair2TermMap nodepair2term,
		const Term2JacobianMaps term2jacobian,
		float *jtj_blks,
		const PenaltyConstants constants = PenaltyConstants())
	{
		const auto nodepair_idx = blockIdx.x;
		const auto encoded_pair = nodepair2term.encoded_nodepair[nodepair_idx];
		const auto term_begin = nodepair2term.nodepair_term_range[nodepair_idx].x;
		const auto term_end = nodepair2term.nodepair_term_range[nodepair_idx].y;
		const auto term_size = term_end - term_begin;
		const auto padded_term_size = warp_size * ((term_size + warp_size - 1) / warp_size);
		const auto lane_id = threadIdx.x % device::warp_size;

		// The memory for store the JtResidual result of each threads
		float local_blks[jtj_blk_size];
		// The memory to perform the reduction
		__shared__ float reduced_blks[jtj_blk_size];

		// Zero out the elements
		for (auto iter = threadIdx.x; iter < jtj_blk_size; iter += warp_size)
			reduced_blks[iter] = 0.0f;
		__syncthreads();

		for (auto iter = threadIdx.x; iter < padded_term_size; iter += warp_size)
		{
			// The global term index
			bool term_valid = true;

			if (iter < term_size)
			{
				const auto term_idx = nodepair2term.nodepair_term_index[term_begin + iter];
				unsigned typed_term_idx;
				TermType term_type;
				query_typed_index(term_idx, nodepair2term.term_offset, term_type, typed_term_idx);

				switch (term_type)
				{
				case TermType::DenseImage:
				{
					// NOTE: For dense image term k, J_i & J_j only has a constant difference
					float jacobian_channelled[d_dense_image_residual_dim * d_node_variable_dim] = {0};
					float nodepair_weight = 0.f;
					computeDenseImageJtJBlockJacobian(term2jacobian.dense_image_term, encoded_pair, typed_term_idx, jacobian_channelled, &nodepair_weight);
					// fillScalarJtJToLocalBlock(term_jacobian, local_blks, constants.DenseImageDepthSquared() * nodepair_weight);
					fillThreeChannelledJtJToLocalBlock(
						jacobian_channelled,
						local_blks,
						constants.DenseImageSquaredVec() * nodepair_weight);
				}
				break;
				case TermType::Reg:
					computeRegJtJLocalBlock(term2jacobian.node_graph_reg_term, typed_term_idx, encoded_pair, local_blks, constants.RegSquared());
					break;
				case TermType::NodeTranslation:
				{
					// Do nothing here, NodeTranslation is a single node regulation
				}
				break;
				case TermType::Feature:
				{
					float term_jacobian[d_node_variable_dim] = {0};
					float nodepair_weight = 0.f;
					// FIXME: to be added
					fillScalarJtJToLocalBlock(term_jacobian, local_blks, constants.FeatureSquared() * nodepair_weight);
				}
				break;
				default:
					term_valid = false;
				}
			}

			//__syncthreads();

			// Do a reduction
			for (int i = 0; i < jtj_blk_size; i++)
			{
				float data = (iter < term_size && term_valid) ? local_blks[i] : 0.0f;
				if (nodepair_idx == 0 && data > 0.f)
				{
					const auto term_idx = nodepair2term.nodepair_term_index[term_begin + iter];
					unsigned typed_term_idx;
					TermType term_type;
					const auto encoded_pair = nodepair2term.encoded_nodepair[nodepair_idx];
					unsigned node_i, node_j;
					const auto term_begin = nodepair2term.nodepair_term_range[nodepair_idx].x;
					const auto term_end = nodepair2term.nodepair_term_range[nodepair_idx].y;
					const auto term_size = term_end - term_begin;
					decode_nodepair(encoded_pair, node_i, node_j);
					query_typed_index(term_idx, nodepair2term.term_offset, term_type, typed_term_idx);
					// printf("%d| thread-idx: %d, term-size: %d, term-idx: %d, data: %f, type: %d, typed_term_idx: %d, pair: (%d, %d)\n",
					//	term_begin + iter, threadIdx.x, term_size, term_idx, data, term_type, typed_term_idx, node_i, node_j);
				}
				data = warp_scan(data);
				if (lane_id == warpSize - 1)
				{
					reduced_blks[i] += data;
				}
			}
		}

		// Write to output
		for (auto iter = threadIdx.x; iter < jtj_blk_size; iter += warp_size)
		{
			jtj_blks[jtj_blk_size * nodepair_idx + iter] = reduced_blks[iter];
		}
	}

	// For performance test
	__global__ void memoryPerformanceTestKernel(
		GArraySlice<float> jtj_blks)
	{
		const auto nodepair_idx = blockIdx.x;
		// Write to output
		for (auto iter = threadIdx.x; iter < jtj_blk_size; iter += warp_size)
			jtj_blks[jtj_blk_size * nodepair_idx + iter] = nodepair_idx;
	}

}

star::JtJMaterializer::JtJMaterializer()
{
	memset(&m_node2term_map, 0, sizeof(m_node2term_map));
	memset(&m_nodepair2term_map, 0, sizeof(m_nodepair2term_map));
	memset(&m_term2jacobian_map, 0, sizeof(m_term2jacobian_map));
}

void star::JtJMaterializer::AllocateBuffer()
{
	m_nondiag_blks.AllocateBuffer(size_t(d_node_variable_dim_square) * size_t(Constants::kMaxNumNodePairs));
	m_binblock_csr_data.AllocateBuffer(size_t(d_node_variable_dim_square) * (size_t(Constants::kMaxNumNodePairs) + size_t(Constants::kMaxNumNodes)));
	m_spmv_handler = std::make_shared<ApplySpMVBinBlockCSR<d_node_variable_dim>>();
}

void star::JtJMaterializer::ReleaseBuffer()
{
	m_nondiag_blks.ReleaseBuffer();
}

void star::JtJMaterializer::SetInputs(
	NodePair2TermMap nodepair2term,
	DenseImageTerm2Jacobian dense_image_term,
	NodeGraphRegTerm2Jacobian node_graph_reg_term,
	NodeTranslationTerm2Jacobian node_translation_term,
	FeatureTerm2Jacobian feature_term,
	Node2TermMap node2term,
	PenaltyConstants constants)
{
	m_nodepair2term_map = nodepair2term;
	m_node2term_map = node2term;

	m_term2jacobian_map.dense_image_term = dense_image_term;
	m_term2jacobian_map.node_graph_reg_term = node_graph_reg_term;
	m_term2jacobian_map.node_translation_term = node_translation_term;
	m_term2jacobian_map.feature_term = feature_term;

	m_penalty_constants = constants;
}

void star::JtJMaterializer::BuildMaterializedJtJNondiagonalBlocks(cudaStream_t stream)
{
	computeNonDiagonalBlocks(stream);
}

void star::JtJMaterializer::computeNonDiagonalBlocks(cudaStream_t stream)
{
	// Correct the size of node pairs
	const auto num_nodepairs = m_nodepair2term_map.encoded_nodepair.Size();
	STAR_CHECK_EQ(num_nodepairs, m_nodepair2term_map.nodepair_term_range.Size());
	m_nondiag_blks.ResizeArrayOrException(num_nodepairs * device::jtj_blk_size);

	// Invoke the kernel
	dim3 blk(device::warp_size);
	dim3 grid(num_nodepairs);
	device::computeJtJNonDiagonalBlockKernel<<<grid, blk, 0, stream>>>(
		m_nodepair2term_map,
		m_term2jacobian_map,
		m_nondiag_blks.Ptr(),
		m_penalty_constants);

	// Sync and check error
#if defined(CUDA_DEBUG_SYNC_CHECK)
	cudaSafeCall(cudaStreamSynchronize(stream));
	cudaSafeCall(cudaGetLastError());
#endif

#ifdef OPT_DEBUG_CHECK
	// Do a sanity check
	nonDiagonalBlocksSanityCheck();
#endif
}