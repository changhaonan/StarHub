#pragma once
#include <star/geometry/geometry_map/GeometryMap.h>
#include <star/geometry/geometry_map/SurfelMap.h>
#include <star/geometry/surfel/SurfelGeometry.h>
#include <star/geometry/surfel/SurfelGeometryInitializer.h>
#include <star/geometry/render/Renderer.h>
#include <star/geometry/node_graph/NodeGraph.h>
#include <star/geometry/keypoint/KeyPoints.h>
#include <star/geometry/surfel/GeometryFusor.h>
// Viewer
#include <easy3d_viewer/context.hpp>

namespace star
{
    class DynamicGeometryProcessor
    {
    public:
        using Ptr = std::shared_ptr<DynamicGeometryProcessor>;
        STAR_NO_COPY_ASSIGN_MOVE(DynamicGeometryProcessor);
        DynamicGeometryProcessor();
        ~DynamicGeometryProcessor();
        void ProcessFrame(
            const SurfelMap& surfel_map,
            const GArrayView<float2> &keypoints,
            const GArrayView<float> &descriptors,
            const GArrayView<DualQuaternion> &solved_se3,
            const unsigned frame_idx,
            cudaStream_t stream);
        void initGeometry(
            const SurfelMap &surfel_map,
            const Eigen::Matrix4f &cam2world,
            const unsigned frame_idx,
            cudaStream_t stream);
        void initKeyPoints(
            const SurfelMap &surfel_map,
            const GArrayView<float2> &keypoints,
            const GArrayView<float> &descriptors,
            const Eigen::Matrix4f &cam2world,
            const unsigned frame_idx,
            cudaStream_t stream);
        void updateGeometry(
            const SurfelMap &surfel_map,
            const GArrayView<DualQuaternion> &solved_se3,
            const unsigned frame_idx,
            cudaStream_t stream);
        void computeSurfelMapTex();
        void computeSurfelMotion(cudaStream_t stream);

        // Access API
        SurfelGeometry::Ptr Geometry(const unsigned frame_idx) const { return m_model_geometry[frame_idx]; };
        KeyPoints::Ptr KeyPoints(const unsigned frame_idx) const { return m_model_keypoints[frame_idx]; };
        NodeGraph::Ptr NodeGraph(const unsigned frame_idx) const { return m_node_graph[frame_idx]; };
        SurfelGeometry::Ptr ActiveGeometry() const { return m_model_geometry[m_buffer_idx]; }
        NodeGraph::Ptr ActiveNodeGraph() const { return m_node_graph[m_buffer_idx]; }
        KeyPoints::Ptr ActiveKeyPoints() const { return m_model_keypoints[m_buffer_idx]; }
        Renderer::SolverMaps GetSolverMaps() const { return m_solver_maps; }
        Renderer::ObservationMaps GetObservationMaps() const { return m_observation_maps; }
        SurfelMapTex GetSurfelMapTex() const { return m_surfel_map_tex; };

        // Visualize
        void saveContext(const unsigned frame_idx, cudaStream_t stream);
        // Render-related
        void drawRenderMaps(const unsigned frame_idx, cudaStream_t stream);

    private:
        // Render-related
        void drawSolverMaps(const unsigned frame_idx, const unsigned geometry_idx, cudaStream_t stream);
        void drawObservationMaps(const unsigned frame_idx, const unsigned geometry_idx, cudaStream_t stream);

        unsigned m_buffer_idx = 0;
        SurfelGeometry::Ptr m_data_geometry;     // Double buffer
        SurfelGeometry::Ptr m_model_geometry[2]; // Geometry
        KeyPoints::Ptr m_model_keypoints[2];     // KeyPoints
        NodeGraph::Ptr m_node_graph[2];
        Renderer::Ptr m_renderer;

        // Vis
        bool m_enable_vis;
        float m_pcd_size;
        float m_node_graph_size;

        // Camera-related
        unsigned m_num_cam;
        Eigen::Matrix4f m_cam2world;
        Intrinsic m_intrinsic;
        unsigned m_img_rows;
        unsigned m_img_cols;
        // Regulation
        floatX<d_max_num_semantic> m_dynamic_regulation;
        // Other
        bool m_enable_semantic_surfel;
        unsigned m_reinit_counter;

        // Flag
        bool m_solver_maps_mapped = false;
        bool m_observation_maps_mapped = false;
        // Map
        Renderer::SolverMaps m_solver_maps;
        Renderer::ObservationMaps m_observation_maps;
        SurfelMapTex m_surfel_map_tex;

        // Operator
        GeometryFusor::Ptr m_geometry_fusor;
    };
}
