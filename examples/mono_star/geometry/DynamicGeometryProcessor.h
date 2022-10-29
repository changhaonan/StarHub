#pragma once
#include <star/geometry/geometry_map/GeometryMap.h>
#include <star/geometry/surfel/SurfelGeometry.h>
#include <star/geometry/surfel/SurfelGeometryInitializer.h>
#include <star/geometry/render/Renderer.h>
#include <star/geometry/node_graph/NodeGraph.h>
#include <mono_star/common/ThreadProcessor.h>
#include <mono_star/common/StarStageBuffer.h>

// Viewer
#include <easy3d_viewer/context.hpp>

namespace star
{
    class DynamicGeometryProcessor : public ThreadProcessor
    {
    public:
        using Ptr = std::shared_ptr<DynamicGeometryProcessor>;
        STAR_NO_COPY_ASSIGN_MOVE(DynamicGeometryProcessor);
        DynamicGeometryProcessor();
        ~DynamicGeometryProcessor();

        void Process(
            StarStageBuffer &star_stage_buffer_this,
            const StarStageBuffer &star_stage_buffer_prev,
            cudaStream_t stream,
            const unsigned frame_idx) override;
        void processFrame(
            const unsigned frame_idx,
            cudaStream_t stream);
        void initGeometry(
            const SurfelMap &surfel_map,
            const Eigen::Matrix4f &cam2world,
            const unsigned frame_idx,
            cudaStream_t stream);

        // Access API
        SurfelGeometry::Ptr Geometry(const unsigned frame_idx) const
        {
            return m_model_geometry[frame_idx];
        };

        // Visualize
        void saveContext(const unsigned frame_idx, cudaStream_t stream);
    private:
        unsigned m_buffer_idx = 0;
        SurfelGeometry::Ptr m_data_geometry;     // Double buffer
        SurfelGeometry::Ptr m_model_geometry[2]; // Double buffer
        NodeGraph::Ptr m_node_graph[2];
        Renderer::Ptr m_renderer;

        // Camera-related
        Eigen::Matrix4f m_cam2world;
    };
}

        // Flag
    };
}