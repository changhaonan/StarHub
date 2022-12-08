#pragma once
#include <star/geometry/surfel/SurfelGeometry.h>
#include <star/geometry/node_graph/NodeGraph.h>
#include <star/geometry/render/Renderer.h>
#include <star/geometry/surfel/SurfelFusionHandler.h>
#include <star/geometry/surfel/FusionRemainingSurfelMarker.h>
#include <star/geometry/surfel/DynamicGeometryAppendHandler.h>
#include <star/geometry/surfel/GeometryCompactHandler.h>

namespace star
{
    // Fusing geometry with measurement
    // class DynamicGeometryFusor
    // {
    // public:
    //     using Ptr = std::shared_ptr<DynamicGeometryFusor>;
    //     DynamicGeometryFusor();
    //     ~DynamicGeometryFusor();
    //     STAR_NO_COPY_ASSIGN_MOVE(DynamicGeometryFusor);

    // private:
    //     // Operation
    //     void geometryRemoval();
    //     void geometryFusion();
    //     void geometryAppend();

    //     // Operator
    //     Renderer::Ptr m_renderer;
    //     SurfelFusionHandler::Ptr m_surfel_fusion_handler;
    //     FusionRemainingSurfelMarker::Ptr m_fusion_remaining_surfel_marker;
    //     GeometryCompactHandler::Ptr m_geometry_compact_handler;
    //     DynamicGeometryAppendHandler::Ptr m_geometry_append_handler;
    // };
}
