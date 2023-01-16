#pragma once
#include <star/common/Counter.h>
#include <star/geometry/geometry_map/SurfelMap.h>
#include <star/geometry/surfel/SurfelGeometry.h>
#include <star/geometry/node_graph/NodeGraph.h>
#include <star/geometry/render/Renderer.h>
#include <star/geometry/surfel/SurfelFusionHandler.h>
#include <star/geometry/surfel/FusionRemainingSurfelMarker.h>
#include <star/geometry/surfel/DynamicGeometryAppendHandler.h>
#include <star/geometry/surfel/GeometryCompactHandler.h>
#include <star/geometry/render/Renderer.h>

namespace star
{
    // Fusing geometry with measurement
    class GeometryFusor
    {
    public:
        using Ptr = std::shared_ptr<GeometryFusor>;
        GeometryFusor(
            SurfelGeometry::Ptr model_surfel_geometry[2],
            NodeGraph::Ptr node_graph[2],
            Renderer::Ptr renderer,
            const unsigned num_cam,
            const unsigned img_cols,
            const unsigned img_rows,
            Extrinsic& cam2world,
            Intrinsic& intrinsic,
            const bool use_semantic,
            const unsigned reinit_count,
            const floatX<d_max_num_semantic> dynamic_regulation);
        ~GeometryFusor();
        STAR_NO_COPY_ASSIGN_MOVE(GeometryFusor);
        using FusionMaps = Renderer::FusionMaps;

        // Fuse the geometry with measurement, and update the node graph
        void Fuse(
            const unsigned active_buffer_idx,
            const unsigned frame_idx,
            const SurfelMapTex& surfel_map,
            SurfelGeometry::Ptr measure_surfel_geometry,
            cudaStream_t stream);

    private:
        // Operation
        void geometryRemoval(
            const Measure4Fusion &measure4fusion,
            const unsigned current_time,
            unsigned &current_geometry_idx,
            unsigned &current_node_graph_idx,
            cudaStream_t stream);
        void geometryRemovalSurfelWarp(
            const Measure4GeometryRemoval &meaure4geometry_removal,
            const unsigned current_time,
            unsigned &current_geometry_idx,
            unsigned &current_node_graph_idx,
            cudaStream_t stream);
        void geometryFusion(
            const Measure4Fusion &measure4fusion,
            const Segmentation4SemanticFusion &segmentation4semantic_fusion,
            Geometry4GeometryAppend &geometry4geometry_append,
            const unsigned current_time,
            unsigned &current_geometry_idx,
            const bool update_semantic,
            const bool geometry_reinit,
            cudaStream_t stream);
        void geometryAppend(
            const Geometry4GeometryAppend &geometry4geometry_append,
            const unsigned current_time,
            unsigned &current_geometry_idx,
            unsigned &current_node_graph_idx,
            cudaStream_t stream);
        void geometrySkinning(
            const unsigned num_remaining_surfel,
            const bool is_incremental,
            unsigned &current_geometry_idx,
            unsigned &current_node_graph_idx,
            cudaStream_t stream);
        // Fusion map
        void drawFusionMaps(
            const unsigned frame_idx,
            const unsigned geometry_idx,
            cudaStream_t stream);

        // Operator
        SurfelFusionHandler::Ptr m_surfel_fusion_handler;
        FusionRemainingSurfelMarker::Ptr m_fusion_remaining_surfel_marker;
        GeometryCompactHandler::Ptr m_geometry_compact_handler;
        DynamicGeometryAppendHandler::Ptr m_geometry_append_handler;

        // Binding
        Renderer::Ptr m_renderer;
        SurfelGeometry::Ptr m_data_surfel_geometry;
        SurfelGeometry::Ptr m_model_surfel_geometry[2];
        NodeGraph::Ptr m_node_graph[2];
        FusionMaps m_fusion_maps;

        // Camera related
        unsigned m_num_cam;
        Extrinsic m_cam2world[d_max_cam];
        Intrinsic m_intrinsic[d_max_cam];
        unsigned m_img_cols[d_max_cam];
        unsigned m_img_rows[d_max_cam];
        // Flag
        bool m_fusion_maps_mapped;

        // Fusion parameters TODO: to be merged into parameter
        float m_counter_node_outtrack_threshold = 50;
        unsigned m_frozen_time = 5;
        bool m_use_semantic = false;
        floatX<d_max_num_semantic> m_dynamic_regulation;

        // Buffer idx
        unsigned m_active_buffer_idx = 0;

        // Support
        Counter m_geometry_counter;
    };
}