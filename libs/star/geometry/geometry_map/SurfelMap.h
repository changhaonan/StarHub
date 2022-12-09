#pragma once
#include <star/common/common_types.h>
#include <star/common/common_texture_utils.h>
#include <star/geometry/geometry_map/GeometryMap.h>

namespace star
{
    class SurfelMapInitializer;
    struct SurfelMapTex
    {
        cudaTextureObject_t vertex_confid;
        cudaTextureObject_t normal_radius;
        cudaTextureObject_t color_time;
        cudaTextureObject_t rgbd; // (R,G,B,D_inv) scaled to -1~1, D_inv is 0~1
        cudaTextureObject_t index;
        cudaTextureObject_t depth;
        cudaTextureObject_t segmentation;
        unsigned num_valid_surfel;
    };

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
        cudaTextureObject_t DepthReadOnly() const { return m_depth.texture; }
        cudaTextureObject_t SegmentationReadOnly() const { return m_segmentation.texture; }
        cudaSurfaceObject_t VertexConfid() { return m_vertex_confid.surface; }
        cudaSurfaceObject_t NormalRadius() { return m_normal_radius.surface; }
        cudaSurfaceObject_t ColorTime() { return m_color_time.surface; }
        cudaSurfaceObject_t RGBD() { return m_rgbd.surface; }
        cudaSurfaceObject_t Index() { return m_index.surface; }
        cudaSurfaceObject_t Depth() { return m_depth.surface; }
        cudaSurfaceObject_t Segmentation() { return m_segmentation.surface; }
        unsigned NumValidSurfels() const { return m_num_valid_surfel; }
        SurfelMapTex Texture() const
        {
            return {
                m_vertex_confid.texture,
                m_normal_radius.texture,
                m_color_time.texture,
                m_rgbd.texture,
                m_index.texture,
                m_depth.texture,
                m_segmentation.texture,
                m_num_valid_surfel};
        }

    private:
        // Surfel-related
        CudaTextureSurface m_vertex_confid;
        CudaTextureSurface m_normal_radius;
        CudaTextureSurface m_color_time;
        CudaTextureSurface m_rgbd; // (R,G,B,D_inv) scaled to -1~1, D_inv is 0~1
        CudaTextureSurface m_index;
        CudaTextureSurface m_depth;
        CudaTextureSurface m_segmentation;
        unsigned m_num_valid_surfel;
    };
}