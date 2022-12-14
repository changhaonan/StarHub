#include <star/geometry/geometry_map/SurfelMap.h>
#include <star/img_proc/surfel_reliable_filter.h>
#include <star/img_proc/generate_maps.h>

star::SurfelMap::SurfelMap(const unsigned width, const unsigned height) : GeometryMap(width, height)
{
    createFloat4TextureSurface(height, width, m_vertex_confid);
    createFloat4TextureSurface(height, width, m_normal_radius);
    createFloat4TextureSurface(height, width, m_color_time);
    createFloat4TextureSurface(height, width, m_rgbd);
    createFloat1TextureSurface(height, width, m_depth);
    createIndexTextureSurface(height, width, m_index);
    createInt32TextureSurface(height, width, m_segmentation);
}

star::SurfelMap::~SurfelMap()
{
    releaseTextureCollect(m_vertex_confid);
    releaseTextureCollect(m_normal_radius);
    releaseTextureCollect(m_color_time);
    releaseTextureCollect(m_rgbd);
    releaseTextureCollect(m_depth);
    releaseTextureCollect(m_index);
    releaseTextureCollect(m_segmentation);
}