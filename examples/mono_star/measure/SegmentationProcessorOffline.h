#pragma once
#include <star/common/common_types.h>
#include <star/io/VolumeDeformFileFetch.h>
#include <star/geometry/geometry_map/SurfelMap.h>
#include <star/geometry/geometry_map/SurfelMapInitializer.h>
#include <mono_star/common/ConfigParser.h>
#include <star/visualization/Visualizer.h>
#include <star/common/common_texture_utils.h>
// Viewer
#include <easy3d_viewer/context.hpp>

namespace star
{

    class SegmentationProcessorOffline
    {
    public:
        using Ptr = std::shared_ptr<SegmentationProcessorOffline>;
        SegmentationProcessorOffline();
        ~SegmentationProcessorOffline();
        STAR_NO_COPY_ASSIGN_MOVE(SegmentationProcessorOffline);
        void ProcessFrame(
            const SurfelMap::Ptr& surfel_map,
            const unsigned frame_idx,
            cudaStream_t stream);

    private:
        void loadSegmentation(const unsigned frame_idx, cudaStream_t stream);
        void prepareReMap();
        void saveContext(const unsigned frame_idx, cudaStream_t stream);
        unsigned m_start_frame_idx;
		unsigned m_step_frame;
        VolumeDeformFileFetch::Ptr m_fetcher;

        // Buffer data
        cv::Mat m_raw_seg_img;  // Raw
        void* m_raw_seg_img_buff;
        GArray<int> m_g_raw_seg_img;  

        cudaTextureObject_t m_segmentation_ref;  // Scale

        // Buffer-ReID
        GArray<int> m_remap;
        unsigned m_max_label;
        std::vector<int> m_semantic_label;

        // Camera-related
        unsigned m_downsample_img_col;
        unsigned m_downsample_img_row;
        Extrinsic m_cam2world;
        float m_downsample_scale;

        // Vis-related
        bool m_enable_vis;
        float m_pcd_size;
    };

}