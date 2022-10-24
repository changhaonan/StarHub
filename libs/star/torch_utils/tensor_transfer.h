#pragma once
#include <torch/torch.h>
#include <star/common/ArrayView.h>
#include <star/common/ArraySlice.h>

namespace star::nn
{

	/*
	 * \brief asTensor: wrap array as tensor, gpu
	 * Support: float, int, short, char
	 */
	template <typename T>
	torch::Tensor asTensor(const T *src_gpu_array, const unsigned array_size);
	template <typename T>
	torch::Tensor asTensor(const GArrayView<T> &src_array_view);

	/*
	 * \brief asTensor: wrap array as 2D-tensor, gpu,
	 * Support: float4
	 */
	template <typename T>
	torch::Tensor asTensor(const T *src_gpu_array, const unsigned cols, const unsigned rows);
	template <typename T>
	torch::Tensor asTensor(const GArrayView<T> &src_array_view, const unsigned cols, const unsigned rows);

	/*
	 * \brief asTensor: wrap array as 3D-tensor, gpu,
	 * Support: float4, float3
	 */
	template <typename T>
	torch::Tensor asTensor(const T *src_gpu_array, const unsigned batch, const unsigned cols, const unsigned rows);
	template <typename T>
	torch::Tensor asTensor(const GArrayView<T> &src_array_view, const unsigned batch, const unsigned cols, const unsigned rows);

	/*
	 * \brief copyTensorAsync: copy tensor to data_ptr
	 * Note: tensor should be contiguous before copy
	 */
	template <typename T>
	void copyTensorAsync(
		const torch::Tensor &src_tensor, T *tar_gpu_array, const unsigned array_size, cudaStream_t stream = 0);
	template <typename T>
	void copyTensorAsync(
		const torch::Tensor &src_tensor, GArraySlice<T> &tar_array_slice, cudaStream_t stream = 0);
	template <typename T>
	void copyTensorAsync(
		const torch::Tensor &src_tensor,
		cudaArray_t &tar_array_2d,
		const unsigned cols,
		const unsigned rows,
		cudaStream_t stream = 0)
	{
		cudaMemcpy2DToArrayAsync(
			tar_array_2d,
			0,
			0,
			src_tensor.contiguous().data_ptr(),
			sizeof(T) * cols,
			sizeof(T) * cols,
			rows,
			cudaMemcpyDeviceToDevice,
			stream);
	}
}