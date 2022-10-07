#pragma once
#include <star/geometry/tsdf/Tsdf.h>
#include <star/common/surfel_types.h>

namespace star
{
    /**
     * \brief Projecting Tsdf to one camera setting. Computing, corresponding vertex, normal, color.
     * \note: This method has some unknown bug for now.
     */
    void RayCastTsdf(
        const Tsdf &tsdf,
        cudaSurfaceObject_t vertex_confid_raycast,
        const Intrinsic &intrinsic,
        const Eigen::Matrix4f &cam2world,
        const unsigned img_rows,
        const unsigned img_cols,
        cudaStream_t stream = 0);
}