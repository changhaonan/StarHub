#include <star/torch_utils/tensor_transfer.h>
#include <star/torch_utils/torch_model.h>

void deleter(void *arg){};

template <>
torch::Tensor star::nn::asTensor(const float *src_gpu_array, const unsigned array_size)
{
    auto options = torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32);
    return torch::from_blob((void *)src_gpu_array, {array_size}, options);
}

template <>
torch::Tensor star::nn::asTensor(const int *src_gpu_array, const unsigned array_size)
{
    auto options = torch::TensorOptions().device(torch::kCUDA).dtype(torch::kInt32);
    return torch::from_blob((void *)src_gpu_array, {array_size}, options);
}

template <>
torch::Tensor star::nn::asTensor(const short *src_gpu_array, const unsigned array_size)
{
    auto options = torch::TensorOptions().device(torch::kCUDA).dtype(torch::kInt16);
    return torch::from_blob((void *)src_gpu_array, {array_size}, options);
}

template <>
torch::Tensor star::nn::asTensor(const char *src_gpu_array, const unsigned array_size)
{
    auto options = torch::TensorOptions().device(torch::kCUDA).dtype(torch::kUInt8);
    return torch::from_blob((void *)src_gpu_array, {array_size}, options);
}

template <typename T>
torch::Tensor star::nn::asTensor(const GArrayView<T> &src_array_view)
{
    return asTensor(src_array_view.Ptr(), src_array_view.Size());
}

template <>
torch::Tensor star::nn::asTensor(const float4 *src_gpu_array, const unsigned cols, const unsigned rows)
{
    auto options = torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32);
    return torch::from_blob((void *)src_gpu_array, {rows, cols, 4}, options);
}

template <typename T>
torch::Tensor star::nn::asTensor(const GArrayView<T> &src_array_view, const unsigned cols, const unsigned rows)
{
    assert(src_array_view.Size() == cols * rows);
    return asTensor(src_array_view.Ptr(), cols, rows);
}

template <>
torch::Tensor star::nn::asTensor(const float4 *src_gpu_array, const unsigned batch, const unsigned cols, const unsigned rows)
{
    auto options = torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32);
    return torch::from_blob((void *)src_gpu_array, {batch, rows, cols, 4}, options);
}

template <>
torch::Tensor star::nn::asTensor(const float3 *src_gpu_array, const unsigned batch, const unsigned cols, const unsigned rows)
{
    auto options = torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32);
    return torch::from_blob((void *)src_gpu_array, {batch, rows, cols, 3}, options);
}

template <typename T>
torch::Tensor star::nn::asTensor(const GArrayView<T> &src_array_view, const unsigned batch, const unsigned cols, const unsigned rows)
{
    assert(src_array_view.Size() == batch * cols * rows);
    return asTensor(src_array_view.Ptr(), batch, cols, rows);
}

template <>
void star::nn::copyTensorAsync(
    const torch::Tensor &src_tensor, float *tar_gpu_array, const unsigned array_size, cudaStream_t stream)
{
    cudaSafeCall(cudaMemcpyAsync(
        tar_gpu_array, src_tensor.contiguous().data_ptr(), array_size * sizeof(float), cudaMemcpyDeviceToDevice, stream));
}

template <>
void star::nn::copyTensorAsync(
    const torch::Tensor &src_tensor, int *tar_gpu_array, const unsigned array_size, cudaStream_t stream)
{
    try
    {
        cudaSafeCall(cudaMemcpyAsync(
            tar_gpu_array, src_tensor.contiguous().data_ptr(), array_size * sizeof(int), cudaMemcpyDeviceToDevice, stream));
    }
    catch (c10::Error &e)
    {
        std::cout << e.what() << std::endl;
    }
}

template <>
void star::nn::copyTensorAsync(
    const torch::Tensor &src_tensor, short *tar_gpu_array, const unsigned array_size, cudaStream_t stream)
{
    cudaSafeCall(cudaMemcpyAsync(
        tar_gpu_array, src_tensor.contiguous().data_ptr(), array_size * sizeof(short), cudaMemcpyDeviceToDevice, stream));
}

template <>
void star::nn::copyTensorAsync(
    const torch::Tensor &src_tensor, char *tar_gpu_array, const unsigned array_size, cudaStream_t stream)
{
    cudaSafeCall(cudaMemcpyAsync(
        tar_gpu_array, src_tensor.contiguous().data_ptr(), array_size * sizeof(char), cudaMemcpyDeviceToDevice, stream));
}

template <typename T>
void star::nn::copyTensorAsync(
    const torch::Tensor &src_tensor, GArraySlice<T> &tar_array_slice, cudaStream_t stream)
{
    copyTensorAsync(src_tensor, tar_array_slice.Ptr(), tar_array_slice.Size(), stream);
}