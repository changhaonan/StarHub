#pragma once
#include <star/common/common_types.h>
#include <star/common/common_texture_utils.h>
#include <star/geometry/geometry_map/GeometryMap.h>

namespace star
{
    /**
     * \brief Two surfel map, usually from measurement or raycasting
     */
    class SurfelMap : public GeometryMap
    {
    public:
        SurfelMap(const unsigned width, const unsigned height);
        ~SurfelMap();
        bool IsEmpty() { return false; };

        /**
         * \brief Generate normal, index from vertex
         */
        void InitFromRGBDImage(
            const GArrayView<uchar3> color_image,
            const GArrayView<unsigned> depth_image,
            cudaStream_t stream
        );

    private:
        CudaTextureSurface vertex_confid;
        CudaTextureSurface normal_radius;
        CudaTextureSurface color_time;
        CudaTextureSurface rgbd; // (R,G,B,D_inv) scaled to -1~1, D_inv is 0~1
        CudaTextureSurface index;
    };
}