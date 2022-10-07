#pragma once
#include <star/common/common_types.h>
#include <star/common/surfel_types.h>

namespace star
{
    /**
     * \brief: The goal of this method is to filter out those measurement with low confidence under real physical system.
     * There are several considerations:
     * \note: We directly filter on depth measurement
     */
    void filterUnreliableDepth(
        cudaTextureObject_t raw_depth_map,      // (mm) unsigned 16
        cudaSurfaceObject_t filtered_depth_map, // (m) float 32
        const unsigned rows, const unsigned cols,
        const float clip_near, const float clip_far, // (m)
        const Intrinsic &intrinsic,
        cudaStream_t stream = 0);

}