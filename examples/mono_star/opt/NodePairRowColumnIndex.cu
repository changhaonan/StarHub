#include <star/common/device_intrinsics.cuh>
#include <star/pcg_solver/solver_configs.h>
#include <star/opt/solver_encode.h>
#include <mono_star/common/term_offset_types.h>
#include <mono_star/opt/NodePair2TermsIndex.h>
#include <device_launch_parameters.h>
#include <cuda_runtime_api.h>

namespace star::device
{
	// Kernel for computing of the row offset in node_pair array
	__global__ void computeRowOffsetKernel(
		const GArrayView<unsigned> compacted_Iij_key,
		GArraySlice<unsigned> rowoffset_array)
	{
		const auto idx = threadIdx.x + blockIdx.x * blockDim.x;
		if (idx >= compacted_Iij_key.Size())
			return;
		if (idx == 0)
		{ // Note: Bin len can't be 0
			rowoffset_array[0] = 0;
			rowoffset_array[rowoffset_array.Size() - 1] = compacted_Iij_key.Size();
		}
		else
		{
			const auto key_prev = compacted_Iij_key[idx - 1];
			const auto key_this = compacted_Iij_key[idx];
			const auto row_prev = encoded_row(key_prev);
			const auto row_this = encoded_row(key_this);

			if (row_this != row_prev)
			{
				rowoffset_array[row_this] = idx;
			}
		}
	}

	// Kernel for computing the length of each row
	// (both diag and non-diagonal terms)
	__global__ void computeRowBlockLengthKernel(
		const unsigned *__restrict__ rowoffset_array,
		GArraySlice<unsigned> blk_rowlength)
	{
		const auto idx = threadIdx.x + blockDim.x * blockIdx.x;
		if (idx >= blk_rowlength.Size())
			return;
		// Note that the diagonal term is included
		blk_rowlength[idx] = 1 + rowoffset_array[idx + 1] - rowoffset_array[idx];
	}

	__global__ void computeBinLengthKernel(
		const GArrayView<unsigned> rowblk_length,
		GArraySlice<unsigned> valid_bin_length,
		GArraySlice<unsigned> m_binnonzeros)
	{
		// The idx is in [0, d_max_num_bin)
		const auto idx = threadIdx.x + blockDim.x * blockIdx.x;
		if (idx >= valid_bin_length.Size())
			return;

		unsigned bin_length = 0;
		// d_bin_size * idx is the real-matrix begin row
		// so does d_bin_size * idx + d_bin_size - 1 is the ending row
		// For a matrix row, its corresponding blk-row is
		// matrix_row / d_surfel_knn_pair_size
		const unsigned blkrow_begin = d_bin_size * idx / d_node_variable_dim;
		unsigned blkrow_end = (d_bin_size * idx + d_bin_size - 1) / d_node_variable_dim;
		blkrow_end = umin(blkrow_end, rowblk_length.Size() - 1);
		unsigned max_length = 0;
		for (unsigned blkrow_idx = blkrow_begin; blkrow_idx <= blkrow_end; blkrow_idx++)
		{
			max_length = umax(max_length, rowblk_length[blkrow_idx]);
		}

		// From block length to actual element length
		bin_length = d_node_variable_dim * max_length;
		valid_bin_length[idx] = bin_length;
		m_binnonzeros[idx] = d_bin_size * bin_length;
	}

	__global__ void computeBinBlockedCSRRowPtrKernel(
		const unsigned *valid_nonzeros_rowscan,
		GArraySlice<int> csr_rowptr)
	{
		const auto idx = threadIdx.x + blockDim.x * blockIdx.x;
		if (idx >= csr_rowptr.Size())
			return;
		const int bin_row_idx = idx / d_bin_size;
		const int bin_row_offset = idx % d_bin_size;
		csr_rowptr[idx] = bin_row_offset + valid_nonzeros_rowscan[bin_row_idx];
	}

	// The column index for bin-block csr format
	__global__ void computeBinBlockedCSRColPtrKernel(
		const unsigned matrix_size,
		const int *csr_rowptr,
		const unsigned *compacted_nodepair,
		const unsigned *blkrow_offset,
		int *csr_colptr)
	{
		const auto row_idx = threadIdx.x + blockDim.x * blockIdx.x;
		if (row_idx >= matrix_size)
			return;

		// From now, the query on rowptr should be safe
		const auto blkrow_idx = row_idx / d_node_variable_dim;
		const auto data_offset = csr_rowptr[row_idx];
		const auto lane_idx = threadIdx.x & 31;

		// For the diagonal terms
		auto column_idx_offset = (data_offset - lane_idx) / d_node_variable_dim + lane_idx;
		csr_colptr[column_idx_offset] = d_node_variable_dim * blkrow_idx;
		column_idx_offset += d_bin_size;

		// For the non-diagonal terms
		auto Iij_begin = blkrow_offset[blkrow_idx];
		const auto Iij_end = blkrow_offset[blkrow_idx + 1];

		for (; Iij_begin < Iij_end; Iij_begin++, column_idx_offset += d_bin_size)
		{
			const auto Iij_key = compacted_nodepair[Iij_begin];
			const auto blkcol_idx = encoded_col(Iij_key);
			csr_colptr[column_idx_offset] = d_node_variable_dim * blkcol_idx;
		}
	}

}

void star::NodePair2TermsIndex::blockRowOffsetSanityCheck()
{
	// FIXME: What will happen if there are node has no neighbor?? (i.e. node-0, node-1)
	LOG(INFO) << "Sanity check for blocked row offset";

	// Checking of the offset
	std::vector<unsigned> compacted_key, row_offset;
	GArrayView<unsigned> compacted_nodepair(m_symmetric_kv_sorter.valid_sorted_key);
	compacted_nodepair.Download(compacted_key);
	row_offset.clear();
	row_offset.push_back(0);
	for (int i = 1; i < compacted_key.size(); i++)
	{
		int this_row = encoded_row(compacted_key[i]);
		int prev_row = encoded_row(compacted_key[i - 1]);
		if (this_row != prev_row)
		{
			row_offset.push_back(i);
		}
	}
	row_offset.push_back(compacted_key.size());

	// Download the gpu offset
	std::vector<unsigned> row_offset_gpu;
	m_blkrow_offset_array.View().Download(row_offset_gpu);
	STAR_CHECK(row_offset.size() == row_offset_gpu.size());
	for (int i = 0; i < row_offset.size(); i++)
	{
		STAR_CHECK(row_offset[i] == row_offset_gpu[i]);
	}

	LOG(INFO) << "Check done";
}

void star::NodePair2TermsIndex::blockRowLengthSanityCheck()
{
	LOG(INFO) << "Sanity check for blocked row length";

	std::vector<unsigned> row_offset;
	m_blkrow_offset_array.View().Download(row_offset);
	STAR_CHECK(row_offset.size() == m_num_nodes + 1);

	// Compute the row size
	std::vector<unsigned> row_length;
	row_length.resize(m_num_nodes);
	for (auto i = 0; i < m_num_nodes; i++)
	{
		row_length[i] = row_offset[i + 1] - row_offset[i] + 1;
	}

	// Download and check the offset
	std::vector<unsigned> row_length_gpu;
	m_blkrow_length_array.View().Download(row_length_gpu);
	STAR_CHECK_EQ(row_length.size(), row_length_gpu.size());
	for (auto i = 0; i < row_length.size(); i++)
	{
		STAR_CHECK(row_length[i] == row_length_gpu[i]);
	}

	LOG(INFO) << "Check done";
}

void star::NodePair2TermsIndex::binLengthNonzerosSanityCheck()
{
	LOG(INFO) << "Sanity check for bin length and non-zeros";

	std::vector<unsigned> bin_length;
	std::vector<unsigned> blk_length;
	m_blkrow_length_array.View().Download(blk_length);
	unsigned num_bins = divUp(d_node_variable_dim * m_num_nodes, d_bin_size);
	bin_length.resize(num_bins);
	for (unsigned i = 0; i < num_bins; i++)
	{
		bin_length[i] = 0;
		for (unsigned row_idx = i * d_bin_size; row_idx < (i + 1) * d_bin_size; row_idx++)
		{
			unsigned blk_row = row_idx / d_node_variable_dim;
			if (blk_row < blk_length.size())
				bin_length[i] = std::max<unsigned>(blk_length[blk_row], bin_length[i]);
		}
		bin_length[i] *= d_node_variable_dim;
	}

	// Download the gpu version for test
	std::vector<unsigned> bin_length_gpu;
	m_binlength_array.View().Download(bin_length_gpu);
	assert(bin_length.size() == bin_length_gpu.size());
	for (size_t i = 0; i < bin_length.size(); i++)
	{
		assert(bin_length[i] == bin_length_gpu[i]);
	}

	// Next check the non-zero values
	std::vector<unsigned> non_zeros;
	non_zeros.resize(bin_length.size() + 1);
	unsigned sum = 0;
	for (size_t i = 0; i < non_zeros.size(); i++)
	{
		non_zeros[i] = sum;
		if (i < bin_length.size())
			sum += d_bin_size * bin_length[i];
	}

	std::vector<unsigned> non_zeros_gpu;
	GArrayView<unsigned>(m_binnonzeros_prefixsum.valid_prefixsum_array).Download(non_zeros_gpu);
	assert(non_zeros.size() == non_zeros_gpu.size());
	for (size_t i = 0; i < non_zeros.size(); i++)
	{
		assert(non_zeros[i] == non_zeros_gpu[i]);
		if (non_zeros[i] != non_zeros_gpu[i])
		{
			std::cout << non_zeros[i] << " " << non_zeros_gpu[i] << std::endl;
		}
	}

	LOG(INFO) << "Check done";
}

void star::NodePair2TermsIndex::binBlockCSRRowPtrSanityCheck()
{
	LOG(INFO) << "Check of the rowptr for bin-blocked csr";

	// Download the data
	std::vector<unsigned> non_zeros;
	GArrayView<unsigned>(m_binnonzeros_prefixsum.valid_prefixsum_array).Download(non_zeros);

	// Check the row-pointer of JtJ
	std::vector<int> JtJ_rowptr_host;
	JtJ_rowptr_host.clear();
	for (int i = 0; i < non_zeros.size(); i++)
	{
		int offset = non_zeros[i];
		for (int j = 0; j < d_bin_size; j++, offset++)
		{
			JtJ_rowptr_host.push_back(offset);
		}
	}
	std::vector<int> JtJ_rowptr_gpu;
	m_binblocked_csr_rowptr.View().Download(JtJ_rowptr_gpu);
	assert(JtJ_rowptr_gpu.size() == JtJ_rowptr_host.size());
	for (int i = 0; i < JtJ_rowptr_host.size(); i++)
	{
		assert(JtJ_rowptr_gpu[i] == JtJ_rowptr_host[i]);
	}

	// Log things here
	// for (auto i = 0; i < 100; ++i) {
	//	std::cout << "Generate row_csr " << i << ": " << JtJ_rowptr_host[i] << std::endl;
	//}

	LOG(INFO) << "Check done";
}

void star::NodePair2TermsIndex::binBlockCSRColumnPtrSanityCheck()
{
	LOG(INFO) << "Sanity check for the colptr of bin-blocked csr format";

	// Prepare the data
	std::vector<int> JtJ_rowptr, JtJ_column_host;
	std::vector<unsigned> blkrow_offset, Iij_compacted_key;
	m_binblocked_csr_rowptr.View().Download(JtJ_rowptr);
	m_blkrow_offset_array.View().Download(blkrow_offset);
	m_symmetric_kv_sorter.valid_sorted_key.download(Iij_compacted_key);

	// Zero out the elements
	GArrayView<int> JtJ_column_index(m_binblocked_csr_colptr.Ptr(), m_binblocked_csr_colptr.BufferSize());
	JtJ_column_host.resize(JtJ_column_index.Size());
	for (int i = 0; i < JtJ_column_host.size(); i++)
	{
		JtJ_column_host[i] = 0;
	}

	for (int row_idx = 0; row_idx < d_node_variable_dim * m_num_nodes; row_idx++)
	{
		int data_offset = JtJ_rowptr[row_idx];
		// First fill the diagonal block
		int blkrow_idx = row_idx / d_node_variable_dim;
		int binwidth_offset = row_idx & 31;
		int column_idx_offset = (data_offset - binwidth_offset) / d_node_variable_dim + binwidth_offset;
		JtJ_column_host[column_idx_offset] = d_node_variable_dim * blkrow_idx;
		for (int i = 0; i < d_node_variable_dim; i++)
		{
			// JtJ_data_host[data_offset] = diag_values[d_bin_size * blkrow_idx + inblk_offset + d_node_variable_dim * i];
			data_offset += d_bin_size;
		}

		// Then fill the non-block values
		column_idx_offset += d_bin_size;
		int key_begin = blkrow_offset[blkrow_idx];
		int key_end = blkrow_offset[blkrow_idx + 1];
		for (int key_iter = key_begin; key_iter < key_end; key_iter++)
		{
			auto Iij_key = Iij_compacted_key[key_iter];
			int Iij_column = encoded_col(Iij_key);
			JtJ_column_host[column_idx_offset] = d_node_variable_dim * Iij_column;
			column_idx_offset += d_bin_size;
			for (int i = 0; i < d_node_variable_dim; i++)
			{
				// JtJ_data_host[data_offset] = nondiag_values[d_bin_size * key_iter + inblk_offset + d_node_variable_dim * i];
				data_offset += d_bin_size;
			}
		}
	}

	// Check the value of column index
	std::vector<int> JtJ_column_gpu;
	JtJ_column_index.Download(JtJ_column_gpu);
	for (int i = 0; i > JtJ_column_host.size(); i++)
	{
		STAR_CHECK(JtJ_column_host[i] == JtJ_column_gpu[i]);
	}

	LOG(INFO) << "Check done";
}

void star::NodePair2TermsIndex::computeBlockRowLength(cudaStream_t stream)
{
	m_blkrow_offset_array.ResizeArrayOrException(m_num_nodes + 1);

	// Prepare the input
	GArrayView<unsigned> compacted_nodepair(m_symmetric_kv_sorter.valid_sorted_key);
	dim3 offset_blk(128);
	dim3 offset_grid(divUp(compacted_nodepair.Size(), offset_blk.x));
	device::computeRowOffsetKernel<<<offset_grid, offset_blk, 0, stream>>>(
		compacted_nodepair,
		m_blkrow_offset_array.Slice());

	// Sync and check error
#if defined(CUDA_DEBUG_SYNC_CHECK)
	cudaSafeCall(cudaStreamSynchronize(stream));
	cudaSafeCall(cudaGetLastError());
#endif

	// Compute the row length
	m_blkrow_length_array.ResizeArrayOrException(m_num_nodes);
	dim3 length_blk(64);
	dim3 length_grid(divUp(m_num_nodes, length_blk.x));
	device::computeRowBlockLengthKernel<<<length_grid, length_blk, 0, stream>>>(
		m_blkrow_offset_array.View(),
		m_blkrow_length_array.Slice());

	// Sync and check error
#if defined(CUDA_DEBUG_SYNC_CHECK)
	cudaSafeCall(cudaStreamSynchronize(stream));
	cudaSafeCall(cudaGetLastError());
#endif

	// Debug sanity check
#ifdef OPT_DEBUG_CHECK
	blockRowOffsetSanityCheck();
	blockRowLengthSanityCheck();
#endif // OPT_DEBUG_CHECK
}

void star::NodePair2TermsIndex::computeBinLength(cudaStream_t stream)
{
	// Correct the size of the matrix
	const auto matrix_size = m_num_nodes * d_node_variable_dim;
	const auto num_bins = divUp(matrix_size, d_bin_size);
	m_binlength_array.ResizeArrayOrException(num_bins);
	m_binnonzeros.ResizeArrayOrException(num_bins + 1);

	dim3 blk(128);
	dim3 grid(divUp(d_max_num_bin, blk.x));
	device::computeBinLengthKernel<<<grid, blk, 0, stream>>>(
		m_blkrow_length_array.View(),
		m_binlength_array.Slice(),
		m_binnonzeros.Slice());

	// Prefix Sum Sync
	m_binnonzeros_prefixsum.ExclusiveSum(m_binnonzeros.Array(), stream);
	cudaSafeCall(cudaStreamSynchronize(stream));

	// Sync and check error
#if defined(CUDA_DEBUG_SYNC_CHECK)
	cudaSafeCall(cudaStreamSynchronize(stream));
	cudaSafeCall(cudaGetLastError());
#endif

	// The sanity check method
#ifdef OPT_DEBUG_CHECK
	binLengthNonzerosSanityCheck();
#endif
}

void star::NodePair2TermsIndex::computeBinBlockCSRRowPtr(cudaStream_t stream)
{
	// Compute the row pointer in bin-blocked csr format
	m_binblocked_csr_rowptr.ResizeArrayOrException(d_bin_size * m_binnonzeros_prefixsum.valid_prefixsum_array.size());
	dim3 rowptr_blk(128);
	dim3 rowptr_grid(divUp(m_binblocked_csr_rowptr.ArraySize(), rowptr_blk.x));
	device::computeBinBlockedCSRRowPtrKernel<<<rowptr_grid, rowptr_blk, 0, stream>>>(
		m_binnonzeros_prefixsum.valid_prefixsum_array.ptr(),
		m_binblocked_csr_rowptr.Slice());

	// Sync and check error
#if defined(CUDA_DEBUG_SYNC_CHECK)
	cudaSafeCall(cudaStreamSynchronize(stream));
	cudaSafeCall(cudaGetLastError());
#endif

	// Sanity check method
	// binBlockCSRRowPtrSanityCheck();
}

void star::NodePair2TermsIndex::nullifyBinBlockCSRColumePtr(cudaStream_t stream)
{
	// Compute the size to nullify
	const auto total_blk_size = m_symmetric_kv_sorter.valid_sorted_key.size() + m_num_nodes;
	// const auto nullify_size = std::min(7 * total_blk_size, m_binblocked_csr_colptr.BufferSize());

	// Do it
	cudaSafeCall(cudaMemsetAsync(
		m_binblocked_csr_colptr.Ptr(),
		0xFF,
		sizeof(int) * m_binblocked_csr_colptr.BufferSize(),
		stream));
}

void star::NodePair2TermsIndex::computeBinBlockCSRColumnPtr(cudaStream_t stream)
{
	// The compacted full nodepair array
	GArrayView<unsigned> compacted_nodepair(m_symmetric_kv_sorter.valid_sorted_key);
	const auto matrix_size = d_node_variable_dim * m_num_nodes;

	// Do not need to query the size of colptr?
	dim3 colptr_blk(128);
	dim3 colptr_grid(divUp(d_bin_size * m_binlength_array.ArraySize(), colptr_blk.x));
	device::computeBinBlockedCSRColPtrKernel<<<colptr_grid, colptr_blk, 0, stream>>>(
		matrix_size,
		m_binblocked_csr_rowptr.Ptr(),
		compacted_nodepair.Ptr(),
		m_blkrow_offset_array.Ptr(),
		m_binblocked_csr_colptr.Ptr());

	// Sync and check error
#if defined(CUDA_DEBUG_SYNC_CHECK)
	cudaSafeCall(cudaStreamSynchronize(stream));
	cudaSafeCall(cudaGetLastError());
#endif

	// Debug method
#ifdef OPT_DEBUG_CHECK
	binBlockCSRColumnPtrSanityCheck();
#endif // OPT_DEBUG_CHECK
}