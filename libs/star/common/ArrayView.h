#pragma once
#include <vector>
#include <cuda_runtime_api.h>
#include <star/common/macro_utils.h>
#include <star/common/common_types.h>

namespace star
{

	// The array view class, as its name, maintains
	// a non-allocate, read-only access to some array.
	template <typename T>
	class GArrayView
	{
	private:
		const T *m_array;
		size_t m_array_size;

	public:
		// Default copy/assign/move/destruct
		__host__ __device__ GArrayView() : m_array(nullptr), m_array_size(0) {}
		__host__ __device__ GArrayView(const T *arr, size_t start, size_t end)
		{
			m_array_size = end - start;
			m_array = arr + start;
		}
		__host__ __device__ GArrayView(const T *arr, size_t size) : m_array(arr), m_array_size(size) {}
		__host__ __device__ GArrayView(const GArray<T> &array) : m_array(array.ptr()), m_array_size(array.size()) {}

		// Simple interface
		__host__ __device__ size_t Size() const { return m_array_size; }
		__host__ __device__ size_t ByteSize() const { return m_array_size * sizeof(T); }
		__host__ __device__ const T *Ptr() const { return m_array; }
		__host__ __device__ operator const T *() const { return m_array; }

		// The accessing method can only be processed on device
		__device__ const T &operator[](size_t index) const { return m_array[index]; }

		// Download to std::vector, typically for debugging
		__host__ void Download(std::vector<T> &h_vec) const
		{
			h_vec.resize(Size());
			cudaSafeCall(cudaMemcpy(h_vec.data(), m_array, Size() * sizeof(T), cudaMemcpyDeviceToHost));
		}
	};

	// The two dimension case of array view
	template <typename T>
	class GArrayView2D
	{
	private:
		unsigned short m_rows, m_cols;
		unsigned m_byte_step; // Note that the step is always in byte
		const T *m_ptr;

	public:
		__host__ __device__ GArrayView2D() : m_rows(0), m_cols(0), m_byte_step(0), m_ptr(nullptr) {}
		__host__ GArrayView2D(const GArray2D<T> &array2D)
			: m_rows(array2D.rows()), m_cols(array2D.cols()),
			  m_byte_step(array2D.step()), m_ptr(array2D.ptr()) {}
		// The interface
		__host__ __device__ __forceinline__ unsigned short Rows() const { return m_rows; }
		__host__ __device__ __forceinline__ unsigned short Cols() const { return m_cols; }
		__host__ __device__ __forceinline__ unsigned ByteStep() const { return m_byte_step; }
		__host__ __device__ __forceinline__ const T *Ptr() const { return m_ptr; }
		__host__ __device__ __forceinline__ const T *Ptr(int row) const
		{
			return ((const T *)((const char *)(m_ptr) + row * m_byte_step));
		}
		__host__ __device__ __forceinline__ const T &operator()(int row, int col) const
		{
			return Ptr(row)[col];
		}
	};

	template <typename T>
	struct PtrStepView
	{
	private:
		unsigned m_byte_step; // Note that the step is always in byte
		const T *m_ptr;

	public:
		__host__ __device__ PtrStepView() : m_byte_step(0), m_ptr(nullptr) {}
		__host__ __device__ PtrStepView(GArrayView2D<T> arrayView2D)
			: m_byte_step(arrayView2D.ByteStep()), m_ptr(arrayView2D.Ptr()) {}

		__host__ __device__ __forceinline__ const T *Ptr() const { return m_ptr; }
		__host__ __device__ __forceinline__ const T *Ptr(int row) const
		{
			return ((const T *)((const char *)(m_ptr) + row * m_byte_step));
		}
		__host__ __device__ __forceinline__ const T &operator()(int row, int col) const
		{
			return Ptr(row)[col];
		}
	};

};
