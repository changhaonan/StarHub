#pragma once
#include <star/common/common_types.h>
#include <star/common/ArrayView.h>
#include <star/common/GBufferArray.h>
#include <star/pcg_solver/BinBlockCSR.h>
#include <star/pcg_solver/ApplySpMVBase.h>

// The cublas library
#include <cublas_v2.h>

namespace star
{
    template <int BlockDim = 6>
    class BlockPCG
    {
    private:
        // The cublas handle
        cublasHandle_t m_cublas_handle;
        cudaStream_t m_stream;

        // The size of buffer that this solver is prepared
        size_t m_max_matrix_size;

        // The memory allocate on device for solver
        GBufferArray<float> r_, t_, d_, q_, s_;
        float *delta_old_;
        float *delta_new_;
        float *dot_dq_;
        float *delta_pagelock_;
        float *delta_0_pagelock_;

        // Use for efficient spmv operation
        cudaTextureObject_t d_texture_;

        // The method to allocate and relase buffer above
        void allocateBuffer(size_t max_matrix_size);
        void releaseBuffer();

        // The memory maintained outside the solver
        GArrayView<float> m_inv_diagonal_blks;
        GArrayView<float> b_;
        GArraySlice<float> x_;
        size_t m_actual_matrix_size;
        typename ApplySpMVBase<BlockDim>::Ptr m_spmv_handler;

    public:
        using Ptr = std::shared_ptr<BlockPCG>;
        explicit BlockPCG(size_t max_matrix_size, cudaStream_t stream = 0);
        ~BlockPCG();
        STAR_NO_COPY_ASSIGN(BlockPCG);

        // This version use abstract apply sparse mv handler
        // Assuming the setup of spmv is ready and the solver
        // will only invoke the ApplySpMV methods
        bool SetSolverInput(
            GArrayView<float> inv_diag_blks,
            typename ApplySpMVBase<BlockDim>::Ptr spmv_handler,
            GArrayView<float> b,
            GArraySlice<float> x_init,
            size_t actual_size = 0);

        // Update the solver and cublas stream
        void UpdateCudaStream(cudaStream_t stream);

        // Solve A x = b with maximum max_iterations, no converge check
        GArrayView<float> Solve(size_t max_iterations = 10, bool zero_init = true);
        GArrayView<float> SolveNoConvergeCheck(size_t max_iteration, bool zero_init = true);

        // Solve A x = b with converge check
        GArrayView<float> SolveConvergeChecked(
            size_t max_iteration,
            bool zero_init = true,
            float epsilon = 2e-3);

        /* The internal processing method for solver
         */
    private:
        // r <- b - A x; d <- inv_diag r; delta_new <- dot(r, d);
        void Initialize(const GArrayView<float> &x_init);

        // r <- b; x <- 0; d <- inv_diag r; delta_new <- dot(r, d);
        void ZeroIntialize();

        // q <- A d; dot_dq <- dot(d, q);
        void PerformSparseMV();
        void TexturedSparseMV();

        /**
         * \brief The second kernel in Weber et al 2013
         *        alpha <- (*delta_new) / (*dot_dq);
         *        x <- x + alpha d; t <- r - alpha q;
         *        s <- inv_diag t; delta_new <- dot(t, s);
         */
        void PerformSecondUpdate();

        // beta <- delta_new / delta_old; d <- s + beta d;
        void PerformThirdUpdate();
    };

}

#if defined(__CUDACC__)
#include "BlockPCG.cuh"
#endif

// The check functions
namespace star
{
    void hostBlockPCGZeroInit(
        const std::vector<float> &b,
        const std::vector<float> &inv_diags,
        std::vector<float> &h_r,
        std::vector<float> &h_d,
        float &dot_r_d);

}