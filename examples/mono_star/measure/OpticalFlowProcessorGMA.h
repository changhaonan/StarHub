#pragma once
#include <chrono>
#include <opencv2/opencv.hpp>
#include <star/common/Lock.h>
#include <star/common/macro_utils.h>
#include <star/common/common_types.h>
#include <star/common/GBufferArray.h>
#include <star/common/algorithm_types.h>
#include <star/geometry/geometry_map/SurfelMap.h>
#include <star/torch_utils/torch_model.h>
#include <star/torch_utils/app/opticalflow/opticalflow_model.h>
#include <star/torch_utils/tensor_transfer.h>
#include <star/visualization/Visualizer.h>
#include <star/common/common_texture_utils.h>
#include <mono_star/common/ConfigParser.h>

namespace star
{

    /* Use a GMA-rgbd model to predict the opticalflow motion
     */
    class OpticalFlowProcessorGMA
    {
    public:
        using Ptr = std::shared_ptr<OpticalFlowProcessorGMA>;
        STAR_NO_COPY_ASSIGN_MOVE(OpticalFlowProcessorGMA);
        OpticalFlowProcessorGMA();
        ~OpticalFlowProcessorGMA();
        void ProcessFrame(
            const SurfelMapTex &surfel_map_this,
            const SurfelMapTex &surfel_map_prev,
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
        void loadRGBD(
            const cudaTextureObject_t &rgbd_tex_this,
            const cudaTextureObject_t &rgbd_tex_prev,
            cudaStream_t stream);
        // Directly load RGBD
        void loadRGBDDirectly(
            const cudaTextureObject_t &rgbd_tex_this,
            const cudaTextureObject_t &rgbd_tex_prev,
            cudaStream_t stream);
        // Load RGBD with a pre-defined fixed background
        void loadRGBDWithBackground(
            const cudaTextureObject_t &rgbd_tex_this,
            const cudaTextureObject_t &rgbd_tex_prev,
            cudaStream_t stream);
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
        void coldStart(); // We need to run 3 times to make the JIT fully compiled
        void saveContext(const unsigned frame_idx, cudaStream_t stream);

        // Model
        nn::TorchModel::Ptr m_model;

        bool m_use_static_background;
        GArray2D<uchar3> m_background_img; // Background pattern to fill in
        unsigned m_num_cam;
        unsigned m_downsample_img_col;
        unsigned m_downsample_img_row;
        Eigen::Matrix4f m_cam2world;
        Intrinsic m_intrinsic;

        float m_opticalflow_suppress_threshold;

        // Buffer
        GBufferArray<float4> m_rgbd_prev;
        GBufferArray<float4> m_rgbd_this;
        CudaTextureSurface m_opticalflow;
        GBufferArray<float4> m_surfel_motion;

        // Vis
        bool m_enable_vis;
        float m_pcd_size;
    };

}
