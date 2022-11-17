#pragma once
#include <memory>
#include <star/common/common_types.h>

namespace star
{
	class BlockPCG6x6
	{
	public:
		using Ptr = std::shared_ptr<BlockPCG6x6>;

		// Default construct/destruct, no copy/assign/move
		explicit BlockPCG6x6() = default;
		~BlockPCG6x6() = default;
		BlockPCG6x6(const BlockPCG6x6 &) = delete;
		BlockPCG6x6(BlockPCG6x6 &&) = delete;
		BlockPCG6x6 &operator=(const BlockPCG6x6 &) = delete;
		BlockPCG6x6 &operator=(BlockPCG6x6 &&) = delete;

		// Allocate and release buffer explicit
		void AllocateBuffer(const unsigned max_maxtrix_size);
		void ReleaseBuffer();

		// The solver interface
		bool SetSolverInput(
			const GArray<float> &diag_blks,
			const GArray<float> &A_data,
			const GArray<int> &A_colptr,
			const GArray<int> &A_rowptr,
			const GArray<float> &b,
			size_t actual_size = 0);

		GArray<float> Solve(const int max_iters = 10, cudaStream_t stream = 0);
		GArray<float> SolveTextured(const int max_iters = 10, cudaStream_t stream = 0);

	private:
		// The buffer maintained inside this class
		size_t m_max_matrix_size;
		GArray<float> p_buffer_, q_buffer_, r_buffer_, s_buffer_, t_buffer_;
		GArray<float> inv_diag_blk_buffer_, x_buffer_;
		cudaTextureObject_t s_texture_;

		// The buffer from setInput method
		size_t m_actual_matrix_size;
		GArray<float> diag_blks_, A_data_, b_;
		GArray<int> A_colptr_, A_rowptr_;
	};

}