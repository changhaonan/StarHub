#include "torch_cpp/tensor_transfer.h"

//template<typename T>
//void star::nn::copyTensorAsync(
//    const torch::Tensor& src_tensor,
//    cudaArray_t& tar_array_2d,
//    const unsigned cols,
//    const unsigned rows,
//    cudaStream_t stream
//) {
//    cudaMemcpy2DToArrayAsync(
//        tar_array_2d,
//        0,
//        0,
//        src_tensor.data_ptr(),
//        sizeof(T) * cols,
//        cols,
//        rows,
//        cudaMemcpyDeviceToDevice,
//        stream);
//}