/**
 * @file compile_test.cu
 * @author Haonan Chang (chnme40cs@gmail.com)
 * @brief This file is used to test the correctness of the compiler.
 * @version 0.1
 * @date 2022-10-05
 * 
 * @copyright Copyright (c) 2022
 * 
 */
#include <star/common/common_types.h>
#include <star/common/ArrayView.h>
#include <star/common/ArraySlice.h>
#include <star/common/GBufferArray.h>
#include <cuda_runtime_api.h>
#include <iostream>

using namespace star;

int main(int argc, char **argv) {
    // Create ArrayBuffer
    GBufferArray<float4> buffer_array;
    buffer_array.AllocateBuffer(1000);
    auto array_view = buffer_array.View();
    buffer_array.ReleaseBuffer();
}