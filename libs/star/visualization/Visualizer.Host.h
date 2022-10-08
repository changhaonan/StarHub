/**
 * \brief Some visualization function is not getting along good with CUDA. Thus, we compile them directly on cpp
 */
#pragma once
#include <star/common/point_cloud_typedefs.h>

namespace star::visualize
{
    // PCD-related
    void DrawPointCloud(const PointCloud3f_Pointer &point_cloud);
}
