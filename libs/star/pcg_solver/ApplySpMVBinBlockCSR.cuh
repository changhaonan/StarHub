#pragma once
#include <star/common/common_texture_utils.h>
#include <star/pcg_solver/BinBlockCSR.h>
#include <star/pcg_solver/ApplySpMVBinBlockCSR.h>
#include <device_launch_parameters.h>
#include <iostream>

namespace star::device
{

	// The interface for spmv
	template <int BlockDim>
	__global__ void performSparseMVKernel(
		const float *A_data,
		const int *A_rowptr,
		const int *A_colptr,
		const float *d,
		GArraySlice<float> q)
	{
		const auto row_idx = threadIdx.x + blockDim.x * blockIdx.x;
		if (row_idx < q.Size())
		{
			// Compute SparseMV
			const auto spmv = BinBlockCSR<BlockDim>::SparseMV(A_data, A_colptr, A_rowptr, d, row_idx);
			q[row_idx] = spmv;
		}
	}

	template <int BlockDim>
	__global__ void performSparseMVKernel(
		const float *A_data,
		const int *A_rowptr,
		const int *A_colptr,
		cudaTextureObject_t d,
		GArraySlice<float> q)
	{
		const auto row_idx = threadIdx.x + blockDim.x * blockIdx.x;
		if (row_idx < q.Size())
		{
			// Compute SparseMV
			const auto spmv = BinBlockCSR<BlockDim>::SparseMV(A_data, A_colptr, A_rowptr, d, row_idx);
			q[row_idx] = spmv;
		}
	}

	// The interface for init residual
	template <int BlockDim>
	__global__ void initializeResidualKernel(
		const GArrayView<float> b,
		const float *x_init,
		const float *A_data,
		const int *A_rowptr,
		const int *A_colptr,
		float *r)
	{
		// The block that this thread is for
		const auto row_idx = threadIdx.x + blockDim.x * blockIdx.x;
		if (row_idx >= b.Size())
			return;

		// Compute SparseMV
		const auto spmv = BinBlockCSR<BlockDim>::SparseMV(A_data, A_colptr, A_rowptr, x_init, row_idx);

		// Store to global result
		r[row_idx] = b[row_idx] - spmv;
	}

	template <int BlockDim>
	__global__ void initializeResidualKernel(
		const GArrayView<float> b,
		cudaTextureObject_t x_init,
		const float *A_data,
		const int *A_rowptr,
		const int *A_colptr,
		float *r)
	{
		// The block that this thread is for
		const auto row_idx = threadIdx.x + blockDim.x * blockIdx.x;
		if (row_idx >= b.Size())
			return;

		// Compute SparseMV
		const auto spmv = BinBlockCSR<BlockDim>::SparseMV(A_data, A_colptr, A_rowptr, x_init, row_idx);

		// Store to global result
		r[row_idx] = b[row_idx] - spmv;
	}

}

template <int BlockDim>
void star::ApplySpMVBinBlockCSR<BlockDim>::ApplySpMV(GArrayView<float> x, GArraySlice<float> spmv_x, cudaStream_t stream)
{
	// Sanity check
	STAR_CHECK_EQ(x.Size(), matrix_size_);
	STAR_CHECK_EQ(spmv_x.Size(), matrix_size_);

	// Perform spmv
	dim3 spmv_blk(128);
	dim3 spmv_grid(divUp(x.Size(), spmv_blk.x));
	device::performSparseMVKernel<BlockDim><<<spmv_grid, spmv_blk, 0, stream>>>(
		A_data_, A_rowptr_, A_colptr_,
		x.Ptr(),
		spmv_x);

	// Sync and check error
#if defined(CUDA_DEBUG_SYNC_CHECK)
	// cudaSafeCall(cudaStreamSynchronize(stream));
	// cudaSafeCall(cudaGetLastError());
#endif
}

template <int BlockDim>
void star::ApplySpMVBinBlockCSR<BlockDim>::ApplySpMVTextured(
	cudaTextureObject_t x,
	GArraySlice<float> spmv_x,
	cudaStream_t stream)
{
	// simple sanity check
	STAR_CHECK_EQ(spmv_x.Size(), matrix_size_);

	dim3 spmv_blk(128);
	dim3 spmv_grid(divUp(spmv_x.Size(), spmv_blk.x));
	device::performSparseMVKernel<BlockDim><<<spmv_grid, spmv_blk, 0, stream>>>(
		A_data_, A_rowptr_, A_colptr_,
		x,
		spmv_x);

	// Sync and check error
#if defined(CUDA_DEBUG_SYNC_CHECK)
	cudaSafeCall(cudaStreamSynchronize(stream));
	cudaSafeCall(cudaGetLastError());
#endif
}

template <int BlockDim>
void star::ApplySpMVBinBlockCSR<BlockDim>::InitResidual(
	GArrayView<float> x_init,
	GArrayView<float> b,
	GArraySlice<float> residual,
	cudaStream_t stream)
{
	// Sanity check
	STAR_CHECK_EQ(x_init.Size(), matrix_size_);
	STAR_CHECK_EQ(b.Size(), matrix_size_);
	STAR_CHECK_EQ(residual.Size(), matrix_size_);

	dim3 spmv_blk(128);
	dim3 spmv_grid(divUp(b.Size(), spmv_blk.x));
	device::initializeResidualKernel<BlockDim><<<spmv_grid, spmv_blk, 0, stream>>>(
		b,
		x_init,
		A_data_, A_rowptr_, A_colptr_,
		residual);

	// Sync and check error
#if defined(CUDA_DEBUG_SYNC_CHECK)
	cudaSafeCall(cudaStreamSynchronize(stream));
	cudaSafeCall(cudaGetLastError());
#endif
}

template <int BlockDim>
void star::ApplySpMVBinBlockCSR<BlockDim>::InitResidualTextured(
	cudaTextureObject_t x_init,
	GArrayView<float> b,
	GArraySlice<float> residual,
	cudaStream_t stream)
{
	// Sanity check
	STAR_CHECK_EQ(b.Size(), matrix_size_);
	STAR_CHECK_EQ(residual.Size(), matrix_size_);

	dim3 spmv_blk(128);
	dim3 spmv_grid(divUp(b.Size(), spmv_blk.x));
	device::initializeResidualKernel<BlockDim><<<spmv_grid, spmv_blk, 0, stream>>>(
		b,
		x_init,
		A_data_, A_rowptr_, A_colptr_,
		residual);

	// Sync and check error
#if defined(CUDA_DEBUG_SYNC_CHECK)
	cudaSafeCall(cudaStreamSynchronize(stream));
	cudaSafeCall(cudaGetLastError());
#endif
}