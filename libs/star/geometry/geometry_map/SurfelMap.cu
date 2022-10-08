#include <star/geometry/geometry_map/SurfelMap.h>
#include <star/img_proc/surfel_reliable_filter.h>
#include <star/img_proc/generate_maps.h>

star::SurfelMap::SurfelMap(const unsigned width, const unsigned height) : GeometryMap(width, height) {
    createFloat4TextureSurface(height, width, vertex_confid);
    createFloat4TextureSurface(height, width, normal_radius);
    createFloat4TextureSurface(height, width, color_time);
    createFloat4TextureSurface(height, width, rgbd);
    createIndexTextureSurface(height, width, index);
}

star::SurfelMap::~SurfelMap() {
    releaseTextureCollect(vertex_confid);
    releaseTextureCollect(normal_radius);
    releaseTextureCollect(color_time);
    releaseTextureCollect(rgbd);
    releaseTextureCollect(index);
}