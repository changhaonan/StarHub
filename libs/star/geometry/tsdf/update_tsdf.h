#pragma once
#include <star/common/common_types.h>
#include <star/common/surfel_types.h>
#include <star/geometry/tsdf/Tsdf.h>

namespace star
{
    /**
     * \brief Using measurement (depth, color) to update tsdf
     */
    void UpdateTsdf(
        const Tsdf &tsdf_prev,
        Tsdf &tsdf_next,
        const cudaTextureObject_t measure_depth_collect, // (m) float 32
        const Intrinsic &intrinsic,
        const Eigen::Matrix4f &cam2world,
        const unsigned img_rows,
        const unsigned img_cols,
        cudaStream_t stream = 0);

    /**
     * Test depth interpolation
     */
    void InspectInterpolation(
        const Tsdf &tsdf,
        const cudaTextureObject_t measure_depth_collect, // (m) float32
        GBufferArray<float4> &interpolated_depth,
        const Intrinsic &intrinsic,
        const Eigen::Matrix4f &cam2world,
        const unsigned img_rows,
        const unsigned img_cols,
        cudaStream_t stream = 0);

    /**
     * \brief
     * \note SectionInspection
     */
    enum SIDirection
    {
        x_axis,
        y_axis,
        z_axis
    };
    void SectionInspectionForUpdate(
        const Tsdf &tsdf,
        const cudaTextureObject_t measure_depth_collect,
        const Intrinsic &intrinsic,
        const Eigen::Matrix4f &cam2world,
        const unsigned img_rows,
        const unsigned img_cols,
        const SIDirection direction,
        const unsigned section_layer,                 // The layer to inspect
        GBufferArray<float4> &section_inspect_surfel, // Measurement inspect that layer
        cudaStream_t stream = 0);
}