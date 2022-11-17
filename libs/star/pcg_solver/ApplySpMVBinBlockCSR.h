#pragma once
#include <star/common/GBufferArray.h>
#include <star/pcg_solver/ApplySpMVBase.h>

namespace star
{
	template <int BlockDim>
	class ApplySpMVBinBlockCSR : public ApplySpMVBase<BlockDim>
	{
	public:
		using Ptr = std::shared_ptr<ApplySpMVBinBlockCSR>;
		STAR_DEFAULT_CONSTRUCT_DESTRUCT(ApplySpMVBinBlockCSR);

		// The interface for matrix size
		size_t MatrixSize() const override { return matrix_size_; }

		// The interface for spmv
		void ApplySpMV(GArrayView<float> x, GArraySlice<float> spmv_x, cudaStream_t stream = 0) override;
		void ApplySpMVTextured(cudaTextureObject_t x, GArraySlice<float> spmv_x, cudaStream_t stream = 0) override;

		// The interface for init residual
		void InitResidual(
			GArrayView<float> x_init,
			GArrayView<float> b,
			GArraySlice<float> residual,
			cudaStream_t stream = 0) override;
		void InitResidualTextured(
			cudaTextureObject_t x_init,
			GArrayView<float> b,
			GArraySlice<float> residual,
			cudaStream_t stream = 0) override;

		// Set the input
		void SetInputs(const float *A_data, const int *A_rowptr, const int *A_colptr, size_t mat_size)
		{
			this->A_data_ = A_data;
			this->A_rowptr_ = A_rowptr;
			this->A_colptr_ = A_colptr;
			this->matrix_size_ = mat_size;
		}

	private:
		// The matrix elements and size
		const float *A_data_;
		const int *A_rowptr_;
		const int *A_colptr_;
		size_t matrix_size_;
	};

}

#if defined(__CUDACC__)
#include "ApplySpMVBinBlockCSR.cuh"
#endif