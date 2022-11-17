#pragma once
#include <star/common/common_types.h>
#include <star/math/device_mat.h>
#include <cuda_texture_types.h>

namespace star
{
	template <int BlockSize>
	class BinBlockCSR
	{
	public:
		/**
		 * \brief Perform sparse matrix vector product for Bin-Blocked CSR format
		 *        for the given row, on GPU it might parallelized over rows
		 * \param A_data
		 * \param A_colptr
		 * \param A_rowptr
		 * \param x
		 * \param row
		 * \return
		 */
		__host__ __device__ static float SparseMV(
			const float *A_data,
			const int *A_colptr,
			const int *A_rowptr,
			const float *x,
			const int row);

		__device__ static float SparseMV(
			const float *A_data,
			const int *A_colptr,
			const int *A_rowptr,
			cudaTextureObject_t x,
			const int row);
	};

}

#include "BinBlockCSR.cuh"

/*
 * The non-template version of checking functions
 */
namespace star
{
	void checkBinBlock6x6SparseMV();
}