#pragma once
#include <star/common/macro_utils.h>
#include <star/common/GBufferArray.h>
#include <memory>

namespace star
{
	template <int BlockDim>
	class BlockDiagonalPreconditionerInverse
	{
	public:
		using Ptr = std::shared_ptr<BlockDiagonalPreconditionerInverse>;
		BlockDiagonalPreconditionerInverse();
		BlockDiagonalPreconditionerInverse(size_t max_matrix_size);
		~BlockDiagonalPreconditionerInverse();
		STAR_NO_COPY_ASSIGN(BlockDiagonalPreconditionerInverse);
		STAR_DEFAULT_MOVE(BlockDiagonalPreconditionerInverse);

		// Allocate and release the buffer
		void AllocateBuffer(size_t max_matrix_size);
		void ReleaseBuffer();

		// The input interface
		void SetInput(GArrayView<float> diagonal_blks);
		void SetInput(GArray<float> diagonal_blks)
		{
			GArrayView<float> diagonal_blks_view(diagonal_blks);
			SetInput(diagonal_blks_view);
		}

		// The processing and access interface
		void PerformDiagonalInverse(cudaStream_t stream = 0);
		GArrayView<float> InversedDiagonalBlocks() const { return m_inv_diag_blks.View(); }

	private:
		// The buffer for the inverse of diagonal blocks
		GBufferArray<float> m_inv_diag_blks;

		// The input to the preconditioner
		GArrayView<float> m_diagonal_blks;
		size_t m_matrix_size;
	};
}

#include "BlockDiagonalPreconditionerInverse.hpp"
#if defined(__CUDACC__)
#include "BlockDiagonalPreconditionerInverse.cuh"
#endif
