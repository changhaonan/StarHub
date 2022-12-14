#pragma once
#include <star/common/common_types.h>
#include <star/common/common_texture_utils.h>
#include <star/geometry/geometry_map/GeometryMap.h>
#include <star/geometry/geometry_map/SurfelMap.h>

namespace star
{
    /**
     * \brief Two surfel map, usually from measurement or raycasting
     */
    class SurfelMapInitializer
    {
    public:
        using Ptr = std::shared_ptr<SurfelMapInitializer>;

        SurfelMapInitializer(
            const unsigned width, const unsigned height,
            const float clip_near, const float clip_far,
            const float surfel_radius_scale, const Intrinsic &intrinsic);
        ~SurfelMapInitializer();

        void UploadDepthImage(
            const GArrayView<unsigned short> depth_image,
            cudaStream_t stream);

        /**
         * \brief Init filtered_depth, color
         */
        void InitFromRGBDImage(
            const GArrayView<uchar3> color_image,
            const GArrayView<unsigned short> depth_image,
            const float init_time,
            SurfelMap &surfel_map,
            cudaStream_t stream);

        /**
         * \brief Create vertex, normal from depth
         */
        void InitFromVertexNormalDepth(
            SurfelMap &surfel_map,
            const float scale,
            cudaStream_t stream);

    private:
        void computeRawVertexFromDepth(
            cudaSurfaceObject_t vertex_confid_buffer,
            const Intrinsic &intrinsic,
            cudaStream_t stream);
        void filterAndScaleVertex(
            cudaTextureObject_t raw_depth_map,
            cudaTextureObject_t raw_vertex_confid_map,
            cudaSurfaceObject_t filtered_vertex_confid_map,
            const float scale,
            const float clip_near,
            const float clip_far,
            cudaStream_t stream);
        void computeIndexMap(
            cudaTextureObject_t vertex_confid_map,
            cudaSurfaceObject_t index_map,
            unsigned &valid_surfel_num,
            const unsigned index_offset,
            cudaStream_t stream);

        // Parameter
        unsigned m_width;
        unsigned m_height;
        float m_clip_near;
        float m_clip_far;
        float m_surfel_radius_scale;
        Intrinsic m_intrinsic;
        // Surfel-related
        SurfelMap::Ptr m_surfel_map;
        // Depth and Filtered depth
        CudaTextureSurface m_raw_depth_img_collect;      // (mm) unsigned short
        CudaTextureSurface m_filtered_depth_img_collect; // (m) float
        CudaTextureSurface m_raw_vertex_confid;

        unsigned *m_valid_count;
    };
}