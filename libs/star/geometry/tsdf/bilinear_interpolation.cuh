#pragma once
#include <cuda_device_runtime_api.h>

namespace star {

    __device__ __forceinline__
    bool check_interpolate_smooth(
        const float Q00, const float Q10, const float Q11, const float Q01, const float interpolate_gap) {
        if (fabs(Q00 - Q10) > interpolate_gap || fabs(Q00 - Q01) > interpolate_gap ||
            fabs(Q11 - Q10) > interpolate_gap || fabs(Q11 - Q01) > interpolate_gap)  // Exist 0
            return false;  // Exist gap
        else
            return true;
    };

    __device__ __forceinline__
    float bilinear_interpolate(
        const float x, const float y,
        const float Q00, const float Q10, const float Q11, const float Q01) {
        const float R0 = (1.f - x) * Q00 + x * Q10;
        const float R1 = (1.f - x) * Q01 + x * Q11;
        const float P = (1.f - y) * R0 + y * R1;
        return P;
    };
}