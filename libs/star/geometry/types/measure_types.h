#pragma once
#include <star/common/common_types.h>
#include <star/common/common_texture_utils.h>

namespace star
{

    /** \brief Observation is different from measurement.
     * Measure is measured from environment.
     * Observation comes from geometry
     */
    struct ObservationMaps
    {
        using Ptr = std::shared_ptr<ObservationMaps>;
        cudaTextureObject_t rgbd_map[d_max_cam];
        cudaTextureObject_t index_map[d_max_cam];
    };
}