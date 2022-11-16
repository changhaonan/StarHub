#pragma once
#include <star/geometry/geometry_map/GeometryMap.h>
#include <star/geometry/geometry_map/SurfelMap.h>
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
        void computeSurfelMapTex();

        // Access API
        SurfelGeometry::Ptr Geometry(const unsigned frame_idx) const
        {
            return m_model_geometry[frame_idx];
        };
        Renderer::SolverMaps GetSolverMaps() const {
            return m_solver_maps;
        }
        Renderer::ObservationMaps GetObservationMaps() const {
            return m_observation_maps;
        }
        SurfelMapTex GetSurfelMapTex() const { return m_surfel_map_tex; };

        // Visualize
        void saveContext(const unsigned frame_idx, cudaStream_t stream);
        // Render-related
        void drawRenderMaps(
            const unsigned frame_idx,
            cudaStream_t stream
        );
    private:
        // Render-related
        void drawSolverMaps(
            const unsigned frame_idx,
            const unsigned geometry_idx,
            cudaStream_t stream
        );
        void drawObservationMaps(
            const unsigned frame_idx,
            const unsigned geometry_idx,
            cudaStream_t stream
        );

        unsigned m_buffer_idx = 0;
        SurfelGeometry::Ptr m_data_geometry;     // Double buffer
        SurfelGeometry::Ptr m_model_geometry[2]; // Double buffer
        NodeGraph::Ptr m_node_graph[2];
        Renderer::Ptr m_renderer;

        // Vis
        bool m_enable_vis;

        // Camera-related
        Eigen::Matrix4f m_cam2world;

        // Flag
        bool m_solver_maps_mapped = false;
        bool m_observation_maps_mapped = false;
        // Map
        Renderer::SolverMaps m_solver_maps;
        Renderer::ObservationMaps m_observation_maps;
        SurfelMapTex m_surfel_map_tex;
    };
}
