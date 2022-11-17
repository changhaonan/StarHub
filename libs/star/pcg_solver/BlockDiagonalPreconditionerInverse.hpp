#pragma once
#include <star/pcg_solver/BlockDiagonalPreconditionerInverse.h>

template <int BlockDim>
star::BlockDiagonalPreconditionerInverse<BlockDim>::BlockDiagonalPreconditionerInverse()
{
	m_matrix_size = 0;
}

template <int BlockDim>
star::BlockDiagonalPreconditionerInverse<BlockDim>::BlockDiagonalPreconditionerInverse(size_t max_matrix_size)
{
	AllocateBuffer(max_matrix_size);
	m_matrix_size = 0;
}

template <int BlockDim>
star::BlockDiagonalPreconditionerInverse<BlockDim>::~BlockDiagonalPreconditionerInverse()
{
	if (m_inv_diag_blks.BufferSize() > 0)
		m_inv_diag_blks.ReleaseBuffer();
}

template <int BlockDim>
void star::BlockDiagonalPreconditionerInverse<BlockDim>::AllocateBuffer(size_t max_matrix_size)
{
	const auto max_blk_num = divUp(max_matrix_size, BlockDim);
	m_inv_diag_blks.AllocateBuffer(max_blk_num * BlockDim * BlockDim);
}

template <int BlockDim>
void star::BlockDiagonalPreconditionerInverse<BlockDim>::ReleaseBuffer()
{
	m_inv_diag_blks.ReleaseBuffer();
}

template <int BlockDim>
void star::BlockDiagonalPreconditionerInverse<BlockDim>::SetInput(GArrayView<float> diagonal_blks)
{
	// Simple sanity check
	STAR_CHECK(diagonal_blks.Size() % (BlockDim * BlockDim) == 0);

	m_diagonal_blks = diagonal_blks;
	m_matrix_size = diagonal_blks.Size() / BlockDim;
	m_inv_diag_blks.ResizeArrayOrException(diagonal_blks.Size());
}