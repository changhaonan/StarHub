#pragma once
#include <star/common/common_types.h>
#include <star/common/common_utils.h>
#include <star/common/macro_utils.h>
#include <star/common/GBufferArray.h>
#include <star/geometry/surfel/SurfelGeometry.h>
#include <star/geometry/geometry_map/SurfelMap.h>
#include <mono_star/common/ConfigParser.h>

namespace star
{

    class NodeMotionProcessor
    {
    public:
        using Ptr = std::shared_ptr<NodeMotionProcessor>;
        STAR_NO_COPY_ASSIGN_MOVE(NodeMotionProcessor);
        NodeMotionProcessor();
        ~NodeMotionProcessor();

        void ProcessFrame(const SurfelMapTex &surfel_map_this,
                          const SurfelMapTex &surfel_map_prev,
                          cudaTextureObject_t opticalflow,
                          const SurfelGeometry::Geometry4Solver &geometry4solver,
                          const unsigned num_node,
                          const unsigned frame_idx, 
                          cudaStream_t stream);
        void ResetNodeMotion(GArraySlice<float4> node_motion_pred, cudaStream_t stream);

        // Public-API
        GArrayView<float4> GetNodeMotionPred() const { return m_node_motion_pred.View(); }
    private:
        void computeNodeMotionVisible(
            const SurfelMapTex &surfel_map_this,
            const SurfelMapTex &surfel_map_prev,
            cudaTextureObject_t opticalflow,
            const SurfelGeometry::Geometry4Solver &geometry4solver,
            const unsigned num_node,
            cudaStream_t stream);
        GBufferArray<float4> m_node_motion_pred;

        // Camera-setting
        unsigned m_downsample_img_cols;
        unsigned m_downsample_img_rows;
        Extrinsic m_cam2world;
        Intrinsic m_intrinsic;

        // Visualized-related
        bool m_enable_vis;
        float m_pcd_size;
    };
}