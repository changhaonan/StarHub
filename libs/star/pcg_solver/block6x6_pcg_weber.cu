#include <star/common/common_types.h>
#include <star/common/common_utils.h>
#include <star/common/sanity_check.h>
#include <star/common/device_intrinsics.cuh>
#include <star/common/Stream.h>
#include <star/common/Serializer.h>
#include <star/common/BinaryFileStream.h>
#include <star/math/DenseGaussian.h>
#include <star/math/DenseLDLT.h>
#include <star/pcg_solver/solver_configs.h>
#include <star/pcg_solver/block6x6_pcg_weber.h>
#include <star/pcg_solver/BinBlockCSR.h>
#include <star/pcg_solver/BlockPCG6x6.h>
#include <device_launch_parameters.h>
#include <cuda_runtime_api.h>
#include <cub/cub.cuh>
#include <iostream>
#include <boost/filesystem.hpp>

namespace star::device
{
	/**
	 * \brief Perform parallel matrix inverse on 6x6 psd matrix array
	 * \tparam num_threads Each thread process a matrix
	 * \param A Input matrix array, will not be touched
	 * \param A_inversed The output matrix array
	 * \param num_matrix
	 */
	template <int num_threads = 64>
	__global__ void matrix6x6InverseKernel(
		const float *A,
		float *A_inversed,
		int num_matrix)
	{
		// Load the matrix into the shared memory
		__shared__ float factored_matrix[36 * num_threads];
		__shared__ float inversed_matrix[36 * num_threads];

		// The input matrix pointer for this block
		const int blk_matrix_offset = 36 * blockDim.x * blockIdx.x;
		const float *A_this_blk = A + blk_matrix_offset;

		// Cooperative loading
		for (auto k = 0; k < 36; k++)
		{ // There are 36 x num_threads float need to be loaded
			if (blk_matrix_offset + k * num_threads + threadIdx.x < num_matrix * 36)
				factored_matrix[k * num_threads + threadIdx.x] = A_this_blk[k * num_threads + threadIdx.x]; // Each thread loads one element
		}

		// Sync here
		__syncthreads();

		// Call the Gaussian inversion
		float *A_this_thread = &(factored_matrix[36 * threadIdx.x]);
		float *A_inv_this_thread = &(inversed_matrix[36 * threadIdx.x]);
		DenseGaussian<6>::Inverse(A_this_thread, A_inv_this_thread);

		// Sync again
		__syncthreads();

		// Cooperative storing
		float *A_inv_this_blk = A_inversed + blk_matrix_offset;
		for (auto k = 0; k < 36; k++)
		{ // There are 36 x num_threads float need to be loaded
			if (blk_matrix_offset + k * num_threads + threadIdx.x < num_matrix * 36)
				A_inv_this_blk[k * num_threads + threadIdx.x] = inversed_matrix[k * num_threads + threadIdx.x]; // Each thread stores one element
		}
	}

	__device__ float nu_old_blk6x6;
	__device__ float nu_new_blk6x6;
	__device__ float reduce_partials_blk6x6[max_reduce_blocks]; // The maximum number of blocks to perform reduce for dot(a, b)

	/**
	 * \brief r <- b; s <- inv_diag_blks * b; mu_new <- dot(r, s)
	 * \tparam num_warps The FIXED number of warps in this kernel, for reduction
	 * \param b
	 * \param inv_diag_blks
	 * \param r
	 * \param s
	 */
	template <int num_warps = reduce_block_warps>
	__global__ void block6x6InitKernel(
		const PtrSz<const float> b,
		const PtrSz<const float> inv_diag_blks,
		PtrSz<float> r,
		PtrSz<float> s,
		PtrSz<float> x)
	{
		// r <- b; s <- inv_diag_blks * b;
		const int idx = threadIdx.x + blockIdx.x * blockDim.x;

		// The dot product from this row for mu_new <- dot(r, s)
		float dot_this_row = 0.0f;
		if (idx < b.size)
		{
			const int blk_idx = idx / 6;

			// Perform the block matrix vector product
			float s_row = 0.0f;
			for (auto j = 0; j < 6; j++)
			{
				const float mat_value = inv_diag_blks[6 * idx + j];
				const float b_value = b[6 * blk_idx + j];
				s_row += mat_value * b_value;
			}
			const float r_row = b[idx];
			dot_this_row = s_row * r_row;

			// Store the value to s and r
			s[idx] = s_row;
			r[idx] = r_row;
			x[idx] = 0.0f;
		}

		// Warp reduction on dot_this_row
		const int warp_id = threadIdx.x >> 5;
		const int lane_id = threadIdx.x & 31;
		float scanned_dot = dot_this_row;
		scanned_dot = warp_scan(scanned_dot);

		// Store the reduced warp_dot to shared memory for block scan
		__shared__ float warp_dot[num_warps];
		if (lane_id == 31)
			warp_dot[warp_id] = scanned_dot;

		// Perform reduct on the warp_dot
		__syncthreads();
		if (warp_id == 0)
		{
			float warp_dot_reduce = 0.0f;
			if (lane_id < num_warps)
				warp_dot_reduce = warp_dot[lane_id];
			// Do warp scan again
			warp_dot_reduce = warp_scan(warp_dot_reduce);
			// Store to global memory
			if (lane_id == 31)
				reduce_partials_blk6x6[blockIdx.x] = warp_dot_reduce;
		}
	}

	__global__ void block6x6ReducePartialKernel()
	{
		float sum = 0.0f;
		if (threadIdx.x < num_reduce_blocks_6x6)
		{
			sum = reduce_partials_blk6x6[threadIdx.x];
		}

		sum = warp_scan(sum);
		if (threadIdx.x == 31)
		{
			nu_new_blk6x6 = sum; // nu_new <- dot(r, s)
		}
	}

	/* nu_old <- nu_new; q <- A s; alpha <- nu_new / dot(q, s); */
	template <int num_warps = reduce_block_warps>
	__global__ void block6x6PCGKernel_0(
		const PtrSz<const float> A_data,
		const PtrSz<const int> A_colptr,
		const PtrSz<const int> A_rowptr,
		const PtrSz<const float> s,
		PtrSz<float> q)
	{
		const int idx = threadIdx.x + blockIdx.x * blockDim.x;
		if (idx == 0)
		{
			nu_old_blk6x6 = nu_new_blk6x6;
		}

		const int warp_id = threadIdx.x >> 5;
		const int lane_id = threadIdx.x & 31;
		float dot_this_row = 0;

		// Perform a sparse matrix-vector product
		if (idx < s.size)
		{
			int begin = A_rowptr[idx];
			const int end = A_rowptr[idx + bin_size];
			int column_offset = (begin - lane_id) / 6 + lane_id;
			float sp_mv = 0.0f;
			while (begin < end)
			{
				const int colume = A_colptr[column_offset];
				for (auto j = 0; j < 6; j++)
				{
					float mat_data = A_data[begin];
					float s_data = colume >= 0 ? s[colume + j] : 0;
					sp_mv += mat_data * s_data;
					begin += bin_size;
				}

				// Increase the column index
				column_offset += bin_size;
			}

			// The value of this row
			q[idx] = sp_mv;
			dot_this_row = sp_mv * s[idx];
		}

		// Perform warp scan
		float scanned_dot = dot_this_row;
		scanned_dot = warp_scan(scanned_dot);

		// Store the reduced warp_dot to shared memory for block scan
		__shared__ float warp_dot[num_warps];
		if (lane_id == 31)
			warp_dot[warp_id] = scanned_dot;

		// Perform reduct on the warp_dot
		__syncthreads();
		if (warp_id == 0)
		{
			float warp_dot_reduce = 0.0f;
			if (lane_id < num_warps)
				warp_dot_reduce = warp_dot[lane_id];
			// Do warp scan again
			warp_dot_reduce = warp_scan(warp_dot_reduce);
			// Store to global memory
			if (lane_id == 31)
				reduce_partials_blk6x6[blockIdx.x] = warp_dot_reduce;
		}
	}

	template <int num_warps = reduce_block_warps>
	__global__ void block6x6PCGKernel_0(
		const PtrSz<const float> A_data,
		const PtrSz<const int> A_colptr,
		const PtrSz<const int> A_rowptr,
		cudaTextureObject_t s,
		PtrSz<float> q)
	{
		const int idx = threadIdx.x + blockIdx.x * blockDim.x;
		if (idx == 0)
		{
			nu_old_blk6x6 = nu_new_blk6x6;
		}

		const int warp_id = threadIdx.x >> 5;
		const int lane_id = threadIdx.x & 31;
		float dot_this_row = 0;

		// Perform a sparse matrix-vector product
		if (idx < q.size)
		{
			int begin = A_rowptr[idx];
			const int end = A_rowptr[idx + bin_size];
			int column_offset = (begin - lane_id) / 6 + lane_id;
			float sp_mv = 0.0f;
			while (begin < end)
			{
				const int colume = A_colptr[column_offset];
				for (auto j = 0; j < 6; j++)
				{
					const float mat_data = A_data[begin];
					const float s_data = (colume >= 0) ? fetch1DLinear<float>(s, colume + j) : 0.0f;
					sp_mv += mat_data * s_data;
					begin += bin_size;
				}

				// Increase the column index
				column_offset += bin_size;
			}

			// The value of this row
			q[idx] = sp_mv;
			dot_this_row = sp_mv * fetch1DLinear<float>(s, idx);
		}

		// Perform warp scan
		float scanned_dot = dot_this_row;
		scanned_dot = warp_scan(scanned_dot);

		// Store the reduced warp_dot to shared memory for block scan
		__shared__ float warp_dot[num_warps];
		if (lane_id == 31)
			warp_dot[warp_id] = scanned_dot;

		// Perform reduct on the warp_dot
		__syncthreads();
		if (warp_id == 0)
		{
			float warp_dot_reduce = 0.0f;
			if (lane_id < num_warps)
				warp_dot_reduce = warp_dot[lane_id];
			// Do warp scan again
			warp_dot_reduce = warp_scan(warp_dot_reduce);
			// Store to global memory
			if (lane_id == 31)
				reduce_partials_blk6x6[blockIdx.x] = warp_dot_reduce;
		}
	}

	/**
	 * \brief alpha <- nu_new / dot(q, s); x <- x + alpha * s;
	 *        t <- r - alpha * q; p <- M_inv*t; nu_new <- dot(t, p)
	 * \tparam num_warps The FIXED number of warps in this kernel
	 */
	template <int num_warps = reduce_block_warps>
	__global__ void block6x6PCGKernel_1(
		const PtrSz<const float> s,
		const PtrSz<const float> r,
		const PtrSz<const float> q,
		const PtrSz<const float> inv_diag_blks,
		PtrSz<float> x,
		PtrSz<float> t,
		PtrSz<float> p)
	{
		// Each block performs a reduction for alpha = dot(q, s)
		__shared__ float alpha;
		const int warp_id = threadIdx.x >> 5;
		const int lane_id = threadIdx.x & 31;
		float scanned_dot;

		// Perform reduction on warp_0
		if (warp_id == 0)
		{
			scanned_dot = 0.0f;
			if (lane_id < num_reduce_blocks_6x6)
			{
				scanned_dot = reduce_partials_blk6x6[lane_id];
			}
			scanned_dot = warp_scan(scanned_dot);
			if (lane_id == 31)
			{
				alpha = nu_new_blk6x6 / scanned_dot;
			}
		}

		// Do sync to broadcast alpha
		__syncthreads();
		const float alpha_thread = alpha;

		// float alpha_thread = alpha;
		const int idx = threadIdx.x + blockDim.x * blockIdx.x;
		float dot_this_row = 0.0f;
		if (idx < x.size)
		{
			const int blk_idx = idx / 6;

			// Block matrix vector product
			float p_row = 0.0;
			float mat_value, r_value;
			for (auto j = 0; j < 6; j++)
			{
				mat_value = inv_diag_blks[6 * idx + j];
				r_value = r[6 * blk_idx + j] - alpha_thread * q[6 * blk_idx + j];
				p_row += mat_value * r_value;
			}
			p[idx] = p_row; // p <- M_inv * r

			// float r_row = r[idx];
			// float q_row = q[idx];
			const float r_row_new = r[idx] - alpha_thread * q[idx];
			t[idx] = r_row_new;				 // t <- r - alpha * q
			x[idx] += alpha_thread * s[idx]; // x <- x + alpha s
			dot_this_row = p_row * r_row_new;
		}

		// Perform in block reduction on dot(q, s)
		scanned_dot = dot_this_row;
		scanned_dot = warp_scan(scanned_dot);

		// Store the reduced warp_dot to shared memory for block scan
		__shared__ float warp_dot[num_warps];
		if (lane_id == 31)
			warp_dot[warp_id] = scanned_dot;

		__syncthreads();
		if (warp_id == 0)
		{
			float warp_dot_reduce = 0.0f;
			if (lane_id < num_warps)
			{
				warp_dot_reduce = warp_dot[lane_id];
			}
			// Do warp scan again
			warp_dot_reduce = warp_scan(warp_dot_reduce);

			// Store to global memory
			if (lane_id == 31)
				reduce_partials_blk6x6[blockIdx.x] = warp_dot_reduce;
		}
	}

	/**
	 * \brief nu_new <- dot(t, p); beta <- nu_new/nu_old; s <- p + beta s
	 */
	__global__ void block6x6PCGKernel_2(
		const PtrSz<const float> p,
		PtrSz<float> s)
	{
		// Each block perform a reduce to compute beta
		__shared__ float beta;
		const int warp_id = threadIdx.x >> 5;
		const int lane_id = threadIdx.x & 31;

		if (warp_id == 0)
		{
			float dot_reduce = 0.0f;
			if (lane_id < num_reduce_blocks_6x6)
			{
				dot_reduce = reduce_partials_blk6x6[lane_id];
			}
			dot_reduce = warp_scan(dot_reduce);
			if (lane_id == 31)
			{
				if (blockIdx.x == 0)
					nu_new_blk6x6 = dot_reduce;
				beta = dot_reduce / nu_old_blk6x6;

				// Debug code: seems correct
				// printf("Beta from device %f \n", beta);
			}
		}

		// Do sync to broadcast the value of beta
		__syncthreads();
		const float beta_thread = beta;
		const auto idx = threadIdx.x + blockDim.x * blockIdx.x;
		if (idx < p.size)
		{
			s[idx] = p[idx] + beta_thread * s[idx];
		}
	}

};

// The check data path for pcg solver
#if defined(_WIN32)
const std::string pcg_data_path = "C:/Users/wei/Documents/Visual Studio 2015/Projects/star/data/pcg_test/";
#else
const std::string pcg_data_path = "/home/wei/Documents/programs/star/data/pcg_test";
#endif

void star::loadCheckData(
	std::vector<float> &A_data,
	std::vector<int> &A_rowptr,
	std::vector<int> &A_colptr,
	std::vector<float> &b,
	std::vector<float> &diag_blks)
{
	std::string pcg_test_file = "pcg_test.dat";

	// Build the path
	using path = boost::filesystem::path;
	path data_dir(pcg_data_path);
	path pcg_test_path = data_dir / pcg_test_file;
	BinaryFileStream input_fstream(pcg_test_path.string().c_str(), BinaryFileStream::Mode::ReadOnly);
	input_fstream.SerializeRead(&A_data);
	input_fstream.SerializeRead(&A_rowptr);
	input_fstream.SerializeRead(&A_colptr);
	input_fstream.SerializeRead(&b);
	input_fstream.SerializeRead(&diag_blks);
}

void star::loadCheckData(
	std::vector<float> &x,
	std::vector<float> &inv_diag_blks)
{
	// The name of loaded values
	std::string result_name = "pcg_result.dat";

	// Build the path
	using path = boost::filesystem::path;
	path data_dir(pcg_data_path);

	// Load them
	path pcg_result_path = data_dir / result_name;
	BinaryFileStream input_fstream(pcg_result_path.string().c_str(), BinaryFileStream::Mode::ReadOnly);
	input_fstream.SerializeRead(&x);
	input_fstream.SerializeRead(&inv_diag_blks);
}

void star::check6x6DiagBlocksInverse(
	const std::vector<float> &diag_blks,
	const std::vector<float> &inv_diag_blks)
{
	// Upload the input to device
	GArray<float> d_diag_blks, d_diag_inversed;
	d_diag_blks.upload(diag_blks);
	d_diag_inversed.create(d_diag_blks.size());
	const auto num_matrix = diag_blks.size() / 36;

	// Call the kernel
	block6x6_diag_inverse(d_diag_blks, d_diag_inversed, num_matrix);

	// Download and check
	std::vector<float> h_diag_inversed;
	d_diag_inversed.download(h_diag_inversed);

	// The checking code
	float max_relative_err = 0.0f;
	double avg_matrix_value = 0.0;
	for (auto i = 0; i < h_diag_inversed.size(); i++)
	{
		float h_diag_value = h_diag_inversed[i];
		float check_diag_value = inv_diag_blks[i];
		avg_matrix_value += h_diag_value;
		if (std::abs(check_diag_value) > 2e-3)
		{
			if (std::abs((check_diag_value - h_diag_value) / h_diag_value) > max_relative_err)
			{
				max_relative_err = std::abs((check_diag_value - h_diag_value) / h_diag_value);
				LOG(INFO) << "The host value and checked value are " << h_diag_value << " and " << check_diag_value;
			}
		}
	}
	std::cout << "The maximun relative error is " << max_relative_err << std::endl;
	std::cout << "The average matrix entry value is " << (avg_matrix_value / h_diag_inversed.size()) << std::endl;
}

void star::block6x6BuildTripletVector(
	const std::vector<float> &A_data,
	const std::vector<int> &A_rowptr,
	const std::vector<int> &A_colptr,
	const int matrix_size,
	std::vector<Eigen::Triplet<float>> &tripletVec)
{
	// Clear the output vector
	tripletVec.clear();

	// Loop over the bins
	int num_bins = divUp(matrix_size, bin_size);
	for (auto bin_idx = 0; bin_idx < num_bins; bin_idx++)
	{
		int first_row = bin_idx * bin_size;
		int bin_data_offset = A_rowptr[first_row];
		int bin_colptr_offset = bin_data_offset / 6;
		// Loop over the row in this bin
		for (auto j = 0; j < bin_size; j++)
		{
			int row_idx = first_row + j;
			int row_data_offset = bin_data_offset + j;
			int row_colptr_offset = bin_colptr_offset + j;
			int row_data_end = A_rowptr[row_idx + bin_size];
			while (row_data_offset < row_data_end)
			{
				for (auto k = 0; k < 6; k++)
				{
					float data = A_data[row_data_offset];
					int column_idx = A_colptr[row_colptr_offset] + k;
					if (column_idx >= 0 && std::abs(data) > 0.0f)
						tripletVec.push_back(Eigen::Triplet<float>(row_idx, column_idx, data));
					row_data_offset += bin_size;
				}
				row_colptr_offset += bin_size;
			}
		}
	}
}

void star::hostEigenSpMV(
	const std::vector<float> &A_data,
	const std::vector<int> &A_rowptr,
	const std::vector<int> &A_colptr,
	const int matrix_size,
	const std::vector<float> &x,
	std::vector<float> &spmv)
{
	// Transfer to Eigen Vector
	Eigen::VectorXf eigen_x;
	eigen_x.resize(x.size());
	for (auto i = 0; i < x.size(); i++)
	{
		eigen_x(i) = x[i];
	}

	// Perform Sparse MV
	hostEigenSpMV(A_data, A_rowptr, A_colptr, matrix_size, eigen_x, spmv);
}

void star::hostEigenSpMV(
	const std::vector<float> &A_data,
	const std::vector<int> &A_rowptr,
	const std::vector<int> &A_colptr,
	const int matrix_size,
	const Eigen::VectorXf &x,
	std::vector<float> &spmv)
{
	// Build the triplet vector
	std::vector<Eigen::Triplet<float>> tripletVec;
	block6x6BuildTripletVector(A_data, A_rowptr, A_colptr, matrix_size, tripletVec);

	// Build the sparse matrix in Eigen
	Eigen::SparseMatrix<float> matrix;
	matrix.resize(matrix_size, matrix_size);
	matrix.setFromTriplets(tripletVec.begin(), tripletVec.end());

	// Do product and store the result
	Eigen::VectorXf product = matrix * x;
	spmv.resize(x.size());
	for (auto i = 0; i < x.size(); i++)
	{
		spmv[i] = product(i);
	}
}

void star::checkBlock6x6PCGSolver(
	const std::vector<float> &diag_blks,
	const std::vector<float> &A_data,
	const std::vector<int> &A_colptr,
	const std::vector<int> &A_rowptr,
	const std::vector<float> &b,
	std::vector<float> &x)
{
	const auto matrix_size = b.size();
	// Prepare the data for device code
	GArray<float> diag_blks_dev, A_data_dev, b_dev, x_dev;
	GArray<int> A_colptr_dev, A_rowptr_dev;
	diag_blks_dev.upload(diag_blks);
	A_data_dev.upload(A_data);
	A_rowptr_dev.upload(A_rowptr);
	A_colptr_dev.upload(A_colptr);
	b_dev.upload(b);
	x_dev.create(matrix_size);

	// Prepare the aux storage
	GArray<float> inv_diag_dev, p, q, r, s, t;
	inv_diag_dev.create(diag_blks_dev.size());
	p.create(matrix_size);
	q.create(matrix_size);
	r.create(matrix_size);
	s.create(matrix_size);
	t.create(matrix_size);

	// Invoke the solver
	GArray<float> valid_x;
	block6x6_pcg_weber(
		diag_blks_dev,
		A_data_dev,
		A_colptr_dev,
		A_rowptr_dev,
		b_dev,
		x_dev,
		inv_diag_dev,
		p, q, r, s, t,
		valid_x);

	// Solve it with class version
#if defined(CHECK_CLASS_PCG6x6)
	BlockPCG6x6 solver;
	solver.AllocateBuffer(matrix_size);
	solver.SetSolverInput(diag_blks_dev, A_data_dev, A_colptr_dev, A_rowptr_dev, b_dev);
	// valid_x = solver.Solve();
	valid_x = solver.SolveTextured();
#endif

	// Check the solved x
	assert(valid_x.size() == x.size());
	valid_x.download(x);

	std::vector<float> spmv;
	spmv.resize(x.size());
	for (auto row = 0; row < x.size(); row++)
	{
		spmv[row] = BinBlockCSR<6>::SparseMV(A_data.data(), A_colptr.data(), A_rowptr.data(), x.data(), row);
	}
}

void star::block6x6_pcg_weber(
	const GArray<float> &diag_blks,
	const GArray<float> &A_data,
	const GArray<int> &A_colptr,
	const GArray<int> &A_rowptr,
	const GArray<float> &b,
	GArray<float> &x_buffer,
	GArray<float> &inv_diag_blk_buffer,
	GArray<float> &p_buffer,
	GArray<float> &q_buffer,
	GArray<float> &r_buffer,
	GArray<float> &s_buffer,
	GArray<float> &t_buffer,
	GArray<float> &valid_x,
	int max_iters,
	cudaStream_t stream)
{

	// Correct the size of array
	size_t N = b.size();
	GArray<float> inv_diag_blks = GArray<float>(inv_diag_blk_buffer.ptr(), diag_blks.size());
	valid_x = GArray<float>(x_buffer.ptr(), N);
	GArray<float> p = GArray<float>(p_buffer.ptr(), N);
	GArray<float> q = GArray<float>(q_buffer.ptr(), N);
	GArray<float> r = GArray<float>(r_buffer.ptr(), N);
	GArray<float> s = GArray<float>(s_buffer.ptr(), N);
	GArray<float> t = GArray<float>(t_buffer.ptr(), N);

	// Compute the inverse of diag blocks for pre-conditioning
	cudaSafeCall(cudaMemsetAsync(valid_x.ptr(), 0, sizeof(float) * valid_x.size(), stream));
	block6x6_diag_inverse(diag_blks, inv_diag_blks, N / 6, stream);

	// The init kernel
	block6x6_init_kernel(b, inv_diag_blks, r, s, valid_x, stream);

	// The main loop
	for (auto i = 0; i < max_iters; i++)
	{
		block6x6_pcg_kernel_0(A_data, A_colptr, A_rowptr, s, q, stream);
		block6x6_pcg_kernel_1(s, r, q, inv_diag_blks, valid_x, t, p, stream);
		block6x6_pcg_kernel_2(p, s, stream);
		r.swap(t);
	}

	// Sync and check error
#if defined(CUDA_DEBUG_SYNC_CHECK)
	cudaSafeCall(cudaStreamSynchronize(stream));
	cudaSafeCall(cudaGetLastError());
#endif
}

void star::block6x6_pcg_weber(
	const GArray<float> &diag_blks,
	const GArray<float> &A_data,
	const GArray<int> &A_colptr,
	const GArray<int> &A_rowptr,
	const GArray<float> &b,
	GArray<float> &x_buffer,
	GArray<float> &inv_diag_blk_buffer,
	GArray<float> &p_buffer,
	GArray<float> &q_buffer,
	GArray<float> &r_buffer,
	GArray<float> &s_buffer,
	cudaTextureObject_t s_texture,
	GArray<float> &t_buffer,
	GArray<float> &valid_x,
	int max_iters,
	cudaStream_t stream)
{
	// Correct the size of array
	size_t N = b.size();
	GArray<float> inv_diag_blks = GArray<float>(inv_diag_blk_buffer.ptr(), diag_blks.size());
	valid_x = GArray<float>(x_buffer.ptr(), N);
	GArray<float> p = GArray<float>(p_buffer.ptr(), N);
	GArray<float> q = GArray<float>(q_buffer.ptr(), N);
	GArray<float> r = GArray<float>(r_buffer.ptr(), N);
	GArray<float> s = GArray<float>(s_buffer.ptr(), N);
	GArray<float> t = GArray<float>(t_buffer.ptr(), N);

	// Compute the inverse of diag blocks for pre-conditioning
	block6x6_diag_inverse(diag_blks, inv_diag_blks, N / 6, stream);

	// The init kernel
	block6x6_init_kernel(b, inv_diag_blks, r, s, valid_x, stream);

	// The main loop
	for (auto i = 0; i < max_iters; i++)
	{
		block6x6_pcg_kernel_0(A_data, A_colptr, A_rowptr, s_texture, q, stream);
		block6x6_pcg_kernel_1(s, r, q, inv_diag_blks, valid_x, t, p, stream);
		block6x6_pcg_kernel_2(p, s, stream);
		r.swap(t);
	}

	// Sync and check error
#if defined(CUDA_DEBUG_SYNC_CHECK)
	cudaSafeCall(cudaStreamSynchronize(stream));
	cudaSafeCall(cudaGetLastError());
#endif
}

void star::block6x6_diag_inverse(const float *A, float *A_inversed, int num_matrix, cudaStream_t stream)
{
	const int threads_per_blk = 64;
	dim3 blk(threads_per_blk);
	dim3 grid(divUp(num_matrix, blk.x));
	device::matrix6x6InverseKernel<threads_per_blk><<<grid, blk, 0, stream>>>(A, A_inversed, num_matrix);

	// Sync and check error
#if defined(CUDA_DEBUG_SYNC_CHECK)
	cudaSafeCall(cudaStreamSynchronize(stream));
	cudaSafeCall(cudaGetLastError());
#endif
}

/* r <- b; s <- inv_diag_blks; mu_new <- dot(r, s)  */
void star::block6x6_init_kernel(
	const GArray<float> &b,
	const GArray<float> &inv_diag_blks,
	GArray<float> &r,
	GArray<float> &s,
	GArray<float> &x,
	cudaStream_t stream)
{
	dim3 blk(reduce_block_threads);
	// dim3 grid(divUp(b.size(), blk.x));
	dim3 grid(num_reduce_blocks_6x6);
	device::block6x6InitKernel<<<grid, blk, 0, stream>>>(b, inv_diag_blks, r, s, x);

	// Perform a reduction on the global memory
	dim3 reduce_blk(32);
	dim3 reduce_grid(1);
	device::block6x6ReducePartialKernel<<<reduce_grid, reduce_blk, 0, stream>>>();

	// Sync and check error
#if defined(CUDA_DEBUG_SYNC_CHECK)
	cudaSafeCall(cudaStreamSynchronize(stream));
	cudaSafeCall(cudaGetLastError());
#endif
}

/* nu_old <- nu_new; q <- A s; alpha <- nu_old / dot(q, s); */
void star::block6x6_pcg_kernel_0(
	const GArray<float> &A_data,
	const GArray<int> &A_colptr,
	const GArray<int> &A_rowptr,
	const GArray<float> &s,
	GArray<float> &q, cudaStream_t stream)
{
	dim3 blk(reduce_block_threads);
	// dim3 grid(divUp(s.size(), blk.x));
	dim3 grid(num_reduce_blocks_6x6);
	device::block6x6PCGKernel_0<<<grid, blk, 0, stream>>>(A_data, A_colptr, A_rowptr, s, q);

	// Sync and check error
#if defined(CUDA_DEBUG_SYNC_CHECK)
	cudaSafeCall(cudaStreamSynchronize(stream));
	cudaSafeCall(cudaGetLastError());
#endif
}

void star::block6x6_pcg_kernel_0(
	const GArray<float> &A_data,
	const GArray<int> &A_colptr,
	const GArray<int> &A_rowptr,
	cudaTextureObject_t s,
	GArray<float> &q,
	cudaStream_t stream)
{
	dim3 blk(reduce_block_threads);
	// dim3 grid(divUp(s.size(), blk.x));
	dim3 grid(num_reduce_blocks_6x6);
	device::block6x6PCGKernel_0<<<grid, blk, 0, stream>>>(A_data, A_colptr, A_rowptr, s, q);

	// Sync and check error
#if defined(CUDA_DEBUG_SYNC_CHECK)
	cudaSafeCall(cudaStreamSynchronize(stream));
	cudaSafeCall(cudaGetLastError());
#endif
}

/* alpha <- nu_new / dot(q, s); x <- x + alpha * s;
 * t <- r - alpha * q; p <- M_inv*t; nu_new <- dot(t, p) */
void star::block6x6_pcg_kernel_1(
	const GArray<float> &s,
	const GArray<float> &r,
	const GArray<float> &q,
	const GArray<float> &inv_diag_blks,
	GArray<float> &x,
	GArray<float> &t,
	GArray<float> &p,
	cudaStream_t stream)
{
	dim3 blk(reduce_block_threads);
	dim3 grid(num_reduce_blocks_6x6);
	device::block6x6PCGKernel_1<<<grid, blk, 0, stream>>>(s, r, q, inv_diag_blks, x, t, p);
}

void star::block6x6_pcg_kernel_2(
	const GArray<float> &p,
	GArray<float> &s,
	cudaStream_t stream)
{
	dim3 blk(256);
	dim3 grid(divUp(s.size(), blk.x));
	device::block6x6PCGKernel_2<<<grid, blk, 0, stream>>>(p, s);
}

/*
 * Below are the checking subroutines defined for 6x6 pcg solver
 */
void star::checkBlock6x6Init(
	const std::vector<float> &b,
	const std::vector<float> &inv_diags,
	std::vector<float> &h_r,
	std::vector<float> &h_s)
{
	// Prepare the data
	GArray<float> b_dev, d_inv_diags, r, s, x;
	b_dev.upload(b);
	d_inv_diags.upload(inv_diags);
	r.create(b_dev.size());
	s.create(b_dev.size());
	x.create(b_dev.size());

	// Call the function
	block6x6_init_kernel(b_dev, d_inv_diags, r, s, x);

	// Check the value of dot product
	cudaDeviceSynchronize();
	r.download(h_r);
	s.download(h_s);
	float dot_value = 0;
	for (auto i = 0; i < h_s.size(); i++)
	{
		dot_value += h_r[i] * h_s[i];
	}

	// Frist check r == b
	assert(h_r.size() == b.size());
	for (auto i = 0; i < b.size(); i++)
	{
		assert(std::abs(h_r[i] - b[i]) < 1e-4);
	}

	// Check s = inv_diag * b
	for (auto row = 0; row < b.size(); row++)
	{
		int blk_idx = row / 6;
		int inblk_offset = row % 6;
		int diag_offset = 36 * blk_idx;
		int diag_start_idx = diag_offset + 6 * inblk_offset;
		float s_row = 0.0f;
		for (auto j = 0; j < 6; j++)
		{
			s_row += inv_diags[diag_start_idx + j] * b[6 * blk_idx + j];
		}
		assert(std::abs(s_row - h_s[row]) < 1e-4);
	}

	// Compare it with device value
	float dot_device;
	cudaMemcpyFromSymbol(&dot_device, device::nu_new_blk6x6, sizeof(float), 0, cudaMemcpyDeviceToHost);
	if (std::abs((dot_device - dot_value) / dot_value) > 1e-6)
	{
		std::cout << "Relative err in init kernel dot product " << std::abs((dot_device - dot_value) / dot_value) << std::endl;
	}
}

void star::checkBlock6x6Init(
	const std::vector<float> &b,
	const std::vector<float> &inv_diags)
{
	std::vector<float> r, s;
	checkBlock6x6Init(b, inv_diags, r, s);
}

void star::checkBlock6x6Kernel_0(
	const std::vector<float> &A_data,
	const std::vector<int> &A_rowptr,
	const std::vector<int> &A_colptr,
	const std::vector<float> &s,
	// Output for later checking
	std::vector<float> &q_device)
{
	// Prepare the data
	GArray<float> d_A_data, s_dev, q_dev;
	GArray<int> d_A_rowptr, d_A_colptr;
	d_A_data.upload(A_data);
	s_dev.upload(s);
	q_dev.create(s.size());
	d_A_colptr.upload(A_colptr);
	d_A_rowptr.upload(A_rowptr);

	// Call device function
	block6x6_pcg_kernel_0(d_A_data, d_A_colptr, d_A_rowptr, s_dev, q_dev);

	// Perform matrix vector product on host
	const auto matrix_size = s.size();
	std::vector<float> q_host;
	hostEigenSpMV(A_data, A_rowptr, A_colptr, matrix_size, s, q_host);

	// Check q = A s
	q_device.clear();
	q_dev.download(q_device);
	float maximum_relative_err = 0.0f;
	assert(q_device.size() == q_host.size());
	for (auto i = 0; i < q_host.size(); i++)
	{
		float host_value = q_host[i];
		float device_value = q_device[i];
		if (std::abs(host_value - device_value) > 1e-4)
		{
			if (std::abs((host_value - device_value) / host_value) > maximum_relative_err)
			{
				maximum_relative_err = std::abs((host_value - device_value) / host_value);
			}
		}
	}
	std::cout << "The maximum relative error in SpMV " << maximum_relative_err << std::endl;

	// Next check the value of dot product
	float dev_dot_reduce[max_reduce_blocks];
	cudaMemcpyFromSymbol(dev_dot_reduce, device::reduce_partials_blk6x6, sizeof(float) * max_reduce_blocks, 0, cudaMemcpyDeviceToHost);
	float dev_dot = 0.0f;
	for (auto j = 0; j < num_reduce_blocks_6x6; j++)
	{
		dev_dot += dev_dot_reduce[j];
	}

	// Compute the dot prodcut at host
	float h_dot = 0.0f;
	for (auto j = 0; j < q_host.size(); j++)
	{
		h_dot += q_host[j] * s[j];
	}
	assert(std::abs((h_dot - dev_dot) / dev_dot) < 1e-4);
}

void star::checkBlock6x6Kernel_1(
	const std::vector<float> &s,
	const std::vector<float> &r,
	const std::vector<float> &q,
	const std::vector<float> &inv_diag_blks,
	std::vector<float> &x,
	std::vector<float> &t,
	std::vector<float> &p)
{
	// Prepare data for input
	GArray<float> s_dev, r_dev, q_dev, inv_diag_blks_dev, x_dev, t_dev, p_dev;
	s_dev.upload(s);
	r_dev.upload(r);
	q_dev.upload(q);
	inv_diag_blks_dev.upload(inv_diag_blks);
	x_dev.upload(x);
	t_dev.create(x_dev.size());
	p_dev.create(x_dev.size());

	// Compute dot product on host
	float dev_dot_reduce[max_reduce_blocks];
	cudaMemcpyFromSymbol(dev_dot_reduce, device::reduce_partials_blk6x6, sizeof(float) * max_reduce_blocks, 0,
						 cudaMemcpyDeviceToHost);
	float dev_dot = 0.0f;
	for (auto j = 0; j < num_reduce_blocks_6x6; j++)
	{
		dev_dot += dev_dot_reduce[j];
	}

	float dot_s_q = 0.0f;
	for (int j = 0; j < q.size(); j++)
	{
		dot_s_q += q[j] * s[j];
	}

	assert(std::abs((dot_s_q - dev_dot) / dev_dot) < 1e-4);

	// Download nu to compute alpha
	float nu_old_host, nu_new_host;
	cudaMemcpyFromSymbol(&nu_old_host, device::nu_old_blk6x6, sizeof(float), 0, cudaMemcpyDeviceToHost);
	cudaMemcpyFromSymbol(&nu_new_host, device::nu_new_blk6x6, sizeof(float), 0, cudaMemcpyDeviceToHost);
	cudaSafeCall(cudaDeviceSynchronize());
	assert(std::abs(nu_new_host - nu_old_host) < 1e-7);
	const float alpha = nu_old_host / dot_s_q;

	// The value of alpha is correct
	// std::cout << "Alpha from host " << alpha << std::endl;

	// Invoke the device version function
	cudaSafeCall(cudaDeviceSynchronize());
	block6x6_pcg_kernel_1(s_dev, r_dev, q_dev, inv_diag_blks_dev, x_dev, t_dev, p_dev);
	cudaSafeCall(cudaDeviceSynchronize());
	cudaSafeCall(cudaGetLastError());

	// Check x <- x + alpha * s
	for (auto i = 0; i < x.size(); i++)
	{
		x[i] += alpha * s[i];
	}
	std::vector<float> h_x_dev;
	x_dev.download(h_x_dev);
	assert(s.size() == x.size());
	auto max_relative_err = maxRelativeError(h_x_dev, x, 1e-3f);
	if (max_relative_err > 1e-5)
	{
		std::cout << "Max relative err for x <- x + alpha s is " << max_relative_err << std::endl;
	}

	// Check t <- r - alpha * q;
	t.resize(s.size());
	std::vector<float> h_t_dev;
	t_dev.download(h_t_dev);
	for (auto j = 0; j < t.size(); j++)
	{
		t[j] = r[j] - alpha * q[j];
		assert(std::abs(t[j] - h_t_dev[j]) < 1e-4);
	}

	// Check p <- M_inv*t;
	std::vector<float> h_p_dev;
	p_dev.download(h_p_dev);
	p.resize(x.size());
	for (auto row = 0; row < t.size(); row++)
	{
		int blk_idx = row / 6;
		int inblk_offset = row % 6;
		int diag_offset = 36 * blk_idx;
		int diag_start_idx = diag_offset + 6 * inblk_offset;
		float p_row = 0.0f;
		for (auto j = 0; j < 6; j++)
		{
			p_row += inv_diag_blks[diag_start_idx + j] * t[6 * blk_idx + j];
		}
		p[row] = p_row;
	}
	max_relative_err = maxRelativeError(h_p_dev, p, 1e-5);
	if (max_relative_err > 1e-5)
	{
		std::cout << "Relative error for p <- Minv t " << max_relative_err << std::endl;
	}

	// Check for nu_new <- dot(t, p)
	float dot_t_p = 0.0f;
	for (auto j = 0; j < p.size(); j++)
	{
		// dot_t_p += h_t_dev[j] * p[j];
		dot_t_p += t[j] * p[j];
	}

	// Download the result to host
	cudaMemcpyFromSymbol(dev_dot_reduce, device::reduce_partials_blk6x6, sizeof(float) * max_reduce_blocks, 0, cudaMemcpyDeviceToHost);
	dev_dot = 0.0f;
	for (auto j = 0; j < num_reduce_blocks_6x6; j++)
	{
		dev_dot += dev_dot_reduce[j];
	}

	// Compare it
	assert(std::abs((dev_dot - dot_t_p) / dot_t_p) < 1e-4);
}

void star::checkBlock6x6Kernel_2(
	const std::vector<float> &p,
	std::vector<float> &s)
{
	// Prepare for device input
	GArray<float> p_dev, s_dev;
	assert(s.size() == p.size());
	p_dev.upload(p);
	s_dev.upload(s);

	// Compute the beta at host
	float parital_reduce[max_reduce_blocks];
	float nu_old_host;
	cudaMemcpyFromSymbol(&nu_old_host, device::nu_old_blk6x6, sizeof(float), 0, cudaMemcpyDeviceToHost);
	cudaMemcpyFromSymbol(parital_reduce, device::reduce_partials_blk6x6,
						 sizeof(float) * max_reduce_blocks, 0, cudaMemcpyDeviceToHost);
	cudaSafeCall(cudaDeviceSynchronize());
	cudaSafeCall(cudaGetLastError());
	float nu_new_host = 0.0f;
	for (auto j = 0; j < num_reduce_blocks_6x6; j++)
	{
		nu_new_host += parital_reduce[j];
	}
	float beta = nu_new_host / nu_old_host;

	// Debug code, seems correct
	// std::cout << "Beta on host " << beta << std::endl;

	// Invoke the kernel
	block6x6_pcg_kernel_2(p_dev, s_dev);

	// Download the nu_new from device
	float nu_new_device;
	cudaMemcpyFromSymbol(&nu_new_device, device::nu_new_blk6x6, sizeof(float), 0, cudaMemcpyDeviceToHost);
	cudaSafeCall(cudaDeviceSynchronize());
	cudaSafeCall(cudaGetLastError());

	// Check that value: seems correct
	assert(std::abs((nu_new_host - nu_new_device) / nu_new_host) < 1e-4);

	// Check s <- p + beta s: seems correct
	std::vector<float> h_s_dev;
	s_dev.download(h_s_dev);
	for (auto i = 0; i < h_s_dev.size(); ++i)
	{
		s[i] = beta * s[i] + p[i];
	}
	auto relative_err = maxRelativeError(s, h_s_dev, 1e-3f);
	if (relative_err > 1e-4)
	{
		std::cout << "Max relative error in s <- p + beta s " << relative_err << std::endl;
	}
}
