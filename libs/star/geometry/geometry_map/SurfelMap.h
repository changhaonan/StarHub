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

        // Public API
        cudaTextureObject_t VertexConfidReadOnly() const { return m_vertex_confid.texture; }
        cudaTextureObject_t NormalRadiusReadOnly() const { return m_normal_radius.texture; }
        cudaTextureObject_t ColorTimeReadOnly() const { return m_color_time.texture; }
        cudaTextureObject_t RGBDReadOnly() const { return m_rgbd.texture; }
        cudaTextureObject_t IndexReadOnly() const { return m_index.texture; }
        cudaSurfaceObject_t VertexConfid() { return m_vertex_confid.surface; }
        cudaSurfaceObject_t NormalRadius() { return m_normal_radius.surface; }
        cudaSurfaceObject_t ColorTime() { return m_color_time.surface; }
        cudaSurfaceObject_t RGBD() { return m_rgbd.surface; }
        cudaSurfaceObject_t Index() { return m_index.surface; }
        unsigned NumValidSurfels() const { return m_num_valid_surfel; }

    private:
        // Surfel-related
        CudaTextureSurface m_vertex_confid;
        CudaTextureSurface m_normal_radius;
        CudaTextureSurface m_color_time;
        CudaTextureSurface m_rgbd; // (R,G,B,D_inv) scaled to -1~1, D_inv is 0~1
        CudaTextureSurface m_index;
        unsigned m_num_valid_surfel;
    };
}