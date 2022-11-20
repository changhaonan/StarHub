#pragma once
#include <star/common/common_types.h>
#include <star/io/VolumeDeformFileFetch.h>
#include <star/geometry/geometry_map/SurfelMap.h>
#include <star/geometry/geometry_map/SurfelMapInitializer.h>
#include <mono_star/common/ConfigParser.h>
// Viewer
#include <easy3d_viewer/context.hpp>

namespace star
{

    /* Use a GMA-rgbd model to predict the opticalflow motion
     */
    class OpticalFlowProcessorOffline
    {
    public:
        using Ptr = std::shared_ptr<OpticalFlowProcessorOffline>;
        STAR_NO_COPY_ASSIGN_MOVE(OpticalFlowProcessorOffline);
        OpticalFlowProcessorOffline();
        ~OpticalFlowProcessorOffline();
        void ProcessFrame(
            const SurfelMapTex &surfel_map_prev,
            const SurfelMapTex &surfel_map_this,
            const unsigned frame_idx,
            cudaStream_t stream);
        // Public-API
        cudaTextureObject_t GetOpticalFlow() const
        {
            return m_opticalflow.texture;
        }
        GArrayView<float4> GetSurfelMotion() const
        {
            return m_surfel_motion.View();
        }

    private:
        void loadOpticalFlow(const unsigned frame_idx, cudaStream_t stream);
        // Compute surfel motion
        void computeSurfelFlowVisible(
            const SurfelMapTex &surfel_map_prev,
            const SurfelMapTex &surfel_map_this,
            cudaStream_t stream);
        void saveOpticalFlow(
            CudaTextureSurface &opticalflow_texsurf,
            cudaStream_t stream);
        // If the optical flow is smaller than a threshold, suppress it to 0
        void saveOpticalFlowWithFilter(
            CudaTextureSurface &opticalflow_texsurf,
            cudaStream_t stream);
        void saveContext(const unsigned frame_idx, cudaStream_t stream);

        float m_opticalflow_suppress_threshold;

        // Reader-related
        unsigned m_start_frame_idx;
		unsigned m_step_frame;
		VolumeDeformFileFetch::Ptr m_fetcher;

        // Camera-related
        unsigned m_raw_img_col;
        unsigned m_raw_img_row;
        unsigned m_downsample_img_col;
        unsigned m_downsample_img_row;
        Extrinsic m_cam2world;
        Intrinsic m_intrinsic;
        float m_downsample_scale;

        // Buffer
        cv::Mat m_raw_opticalflow_img;  // Raw
        void* m_raw_opticalflow_img_buff;
		GArray<float2> m_g_raw_opticalflow_img;
        CudaTextureSurface m_opticalflow;  // Scaled
        GBufferArray<float4> m_surfel_motion;

        // Vis
        bool m_enable_vis;
        float m_pcd_size;
    };

}
