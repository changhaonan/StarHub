#pragma once
#include <star/common/common_types.h>
#include <star/common/common_texture_utils.h>
#include <star/geometry/geometry_map/GeometryMap.h>

namespace star
{
    class SurfelMapInitializer;

    /**
     * \brief Two surfel map, usually from measurement or raycasting
     */
    class SurfelMap : public GeometryMap
    {
    public:
        friend class SurfelMapInitializer;
        using Ptr = std::shared_ptr<SurfelMap>;
        SurfelMap(const unsigned width, const unsigned height);
        ~SurfelMap();
        bool IsEmpty() { return false; };

    private:
        // Surfel-related
        CudaTextureSurface m_vertex_confid;
        CudaTextureSurface m_normal_radius;
        CudaTextureSurface m_color_time;
        CudaTextureSurface m_rgbd; // (R,G,B,D_inv) scaled to -1~1, D_inv is 0~1
        CudaTextureSurface m_index;
    };
}