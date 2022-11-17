#pragma once
#include <star/math/DenseGaussian.h>
#include <star/pcg_solver/BlockDiagonalPreconditionerInverse.h>
#include <device_launch_parameters.h>

namespace star::device
{

	enum
	{
		thread_size = opt_warp_size
	};

	template <int BlockDim>
	__global__ void blockDiagonalInverseKernel(
		const float *__restrict__ A,
		float *__restrict__ A_inversed,
		const unsigned num_matrix)
	{
		// Load the matrix into the shared memory
		int blk_size = BlockDim * BlockDim;
		extern __shared__ char shared_array[];
		float *factored_matrix = (float *)shared_array;
		float *inversed_matrix = (float *)(blk_size * thread_size * sizeof(float) + shared_array);

		// The input matrix pointer for this block
		const int blk_matrix_offset = blk_size * blockDim.x * blockIdx.x;
		const float *A_this_blk = A + blk_matrix_offset;

		// Cooperative loading
		for (auto k = 0; k < blk_size; k++)
		{
			if (blk_matrix_offset + k * thread_size + threadIdx.x < num_matrix * blk_size)
				factored_matrix[k * thread_size + threadIdx.x] = A_this_blk[k * thread_size + threadIdx.x]; // Each thread loads one element
		}

		// Sync here
		__syncthreads();

		// Call the Gaussian inversion
		float *A_this_thread = &(factored_matrix[blk_size * threadIdx.x]);
		float *A_inv_this_thread = &(inversed_matrix[blk_size * threadIdx.x]);
		DenseGaussian<BlockDim>::Inverse(A_this_thread, A_inv_this_thread);

		// Sync again
		__syncthreads();

		// Cooperative storing
		float *A_inv_this_blk = A_inversed + blk_matrix_offset;
		for (auto k = 0; k < blk_size; k++)
		{
			if (blk_matrix_offset + k * thread_size + threadIdx.x < num_matrix * blk_size)
				A_inv_this_blk[k * thread_size + threadIdx.x] = inversed_matrix[k * thread_size + threadIdx.x]; // Each thread stores one element
		}
	}
}

template <int BlockDim>
void star::BlockDiagonalPreconditionerInverse<BlockDim>::PerformDiagonalInverse(cudaStream_t stream)
{
	const auto num_blks = m_matrix_size / BlockDim;
	dim3 inv_blk(device::thread_size);
	dim3 inv_grid(divUp(num_blks, inv_blk.x));
	size_t shared_mem_size = 2 * device::thread_size * BlockDim * BlockDim * sizeof(float);
	device::blockDiagonalInverseKernel<BlockDim><<<inv_grid, inv_blk, shared_mem_size, stream>>>(
		m_diagonal_blks.Ptr(), m_inv_diag_blks.Ptr(), num_blks);

	// Sync and check error
#if defined(CUDA_DEBUG_SYNC_CHECK)
	cudaSafeCall(cudaStreamSynchronize(stream));
	cudaSafeCall(cudaGetLastError());
#endif
}
