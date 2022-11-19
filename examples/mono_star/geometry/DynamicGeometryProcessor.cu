#include <star/geometry/node_graph/Skinner.h>
#include <star/geometry/surfel/SurfelNodeDeformer.h>
#include <star/visualization/Visualizer.h>
#include <mono_star/common/ConfigParser.h>
#include "DynamicGeometryProcessor.h"

star::DynamicGeometryProcessor::DynamicGeometryProcessor()
{
    auto &config = ConfigParser::Instance();

    m_data_geometry = std::make_shared<star::SurfelGeometry>();
    m_model_geometry[0] = std::make_shared<star::SurfelGeometry>();
    m_model_geometry[1] = std::make_shared<star::SurfelGeometry>();
    m_node_graph[0] = std::make_shared<star::NodeGraph>(config.node_radius());
    m_node_graph[1] = std::make_shared<star::NodeGraph>(config.node_radius());

    // Render
    m_renderer = std::make_shared<star::Renderer>(
        config.num_cam(),
        config.downsample_img_cols(),
        config.downsample_img_rows(),
        config.rgb_intrinsic_downsample(),
        config.max_rendering_depth());

    m_renderer->MapDataSurfelGeometryToCuda(0, *m_data_geometry);
    m_renderer->MapModelSurfelGeometryToCuda(0, *m_model_geometry[0]);
    m_renderer->MapModelSurfelGeometryToCuda(1, *m_model_geometry[1]);

    // Vis
    m_enable_vis = config.enable_vis();
    m_pcd_size = config.pcd_size();

    // Camera-related
    m_cam2world = config.extrinsic()[0];
}

star::DynamicGeometryProcessor::~DynamicGeometryProcessor()
{
    m_renderer->UnmapDataSurfelGeometryFromCuda(0);
    m_renderer->UnmapModelSurfelGeometryFromCuda(0);
    m_renderer->UnmapModelSurfelGeometryFromCuda(1);

    if (m_solver_maps_mapped)
        m_renderer->UnmapSolverMapsFromCuda();
    if (m_observation_maps_mapped)
        m_renderer->UnmapObservationMapsFromCuda();
}

void star::DynamicGeometryProcessor::ProcessFrame(
    const GArrayView<DualQuaternion> &solved_se3,
    const unsigned frame_idx,
    cudaStream_t stream)
{
    if (frame_idx > 0)
    { // Can apply warp
        updateGeometry(solved_se3, frame_idx, stream);
    }

    // Generate map from geometry
    drawRenderMaps(frame_idx, stream);
    computeSurfelMapTex();

    // Vis
    if (m_enable_vis)
        saveContext(frame_idx, stream);
}

void star::DynamicGeometryProcessor::initGeometry(
    const SurfelMap &surfel_map, const Eigen::Matrix4f &cam2world, const unsigned frame_idx, cudaStream_t stream)
{
    // Update buffer_idx
    // m_buffer_idx = (m_buffer_idx + 1) % 2;

    // Init Surfel geometry
    SurfelGeometryInitializer::InitFromGeometryMap(
        *m_model_geometry[m_buffer_idx],
        surfel_map,
        cam2world,
        stream);

    // Init NodeGraph
    m_node_graph[m_buffer_idx]->InitializeNodeGraphFromVertex(
        m_model_geometry[m_buffer_idx]->LiveVertexConfidenceReadOnly(), frame_idx, false, stream);
    m_node_graph[m_buffer_idx]->ResetNodeGraphConnection(stream);

    // Init Skinning
    auto geometyr4skinner = m_model_geometry[m_buffer_idx]->GenerateGeometry4Skinner();
    auto node_graph4skinner = m_node_graph[m_buffer_idx]->GenerateNodeGraph4Skinner();
    Skinner::PerformSkinningFromRef(geometyr4skinner, node_graph4skinner, stream);
}

void star::DynamicGeometryProcessor::updateGeometry(
    const GArrayView<DualQuaternion> &solved_se3,
    const unsigned frame_idx,
    cudaStream_t stream)
{
    if (solved_se3.Size() == 0)
        return;
        
    // Apply the deformation
    SurfelNodeDeformer::ForwardWarpSurfelsAndNodes(m_node_graph[m_buffer_idx]->DeformAccess(),
                                                   *m_model_geometry[m_buffer_idx], solved_se3, stream);

    // Reanchor the geometry
    auto next_buffer_idx = (m_buffer_idx + 1) & 1;
    SurfelGeometry::ReAnchor(
        m_model_geometry[m_buffer_idx],
        m_model_geometry[next_buffer_idx],
        stream);
    NodeGraph::ReAnchor(
        m_node_graph[m_buffer_idx],
        m_node_graph[next_buffer_idx],
        stream);
    m_buffer_idx = next_buffer_idx;
}

void star::DynamicGeometryProcessor::computeSurfelMapTex()
{
    // Get solver-map
    m_surfel_map_tex.vertex_confid = m_solver_maps.vertex_confid_map[0];
    m_surfel_map_tex.normal_radius = m_solver_maps.normal_radius_map[0];

    // Get observation-map
    m_surfel_map_tex.rgbd = m_observation_maps.rgbd_map[0];
    m_surfel_map_tex.index = m_observation_maps.index_map[0];
    m_surfel_map_tex.color_time = 0;

    // Num
    m_surfel_map_tex.num_valid_surfel = m_model_geometry[m_buffer_idx]->NumValidSurfels();
}

void star::DynamicGeometryProcessor::saveContext(const unsigned frame_idx, cudaStream_t stream)
{
    auto &context = easy3d::Context::Instance();

    // Save Geometry
    unsigned last_buffer_idx = (m_buffer_idx + 1) % 2;
    std::string last_geo_name = "last_vertex";
    if (m_model_geometry[last_buffer_idx]->NumValidSurfels() > 0)
    {
        // Exist valid last geo
        std::string ref_geo_name = "ref_geo";
        context.addPointCloud(ref_geo_name, ref_geo_name, m_cam2world.inverse(), m_pcd_size);
        visualize::SavePointCloud(
            m_model_geometry[last_buffer_idx]->ReferenceVertexConfidenceReadOnly(),
            context.at(ref_geo_name));

        std::string live_geo_name = "live_geo";
        context.addPointCloud(live_geo_name, live_geo_name, m_cam2world.inverse(), m_pcd_size);
        visualize::SavePointCloud(
            m_model_geometry[last_buffer_idx]->LiveVertexConfidenceReadOnly(),
            context.at(live_geo_name));

        // Save node graph
        context.addGraph("ref_graph", "", m_cam2world.inverse(), m_pcd_size);
        visualize::SaveGraph(
            m_node_graph[last_buffer_idx]->GetReferenceNodeCoordinate(),
            m_node_graph[last_buffer_idx]->GetNodeKnn(),
            context.at("ref_graph"));

        context.addGraph("live_graph", "", m_cam2world.inverse(), m_pcd_size);
        visualize::SaveGraph(
            m_node_graph[last_buffer_idx]->GetLiveNodeCoordinate(),
            m_node_graph[last_buffer_idx]->GetNodeKnn(),
            context.at("live_graph"));
    }

    // Save images
    context.addImage("ref-rgb");
    context.addImage("ref-depth");
    visualize::SaveNormalizeRGBDImage(m_observation_maps.rgbd_map[0], context.at("ref-rgb"), context.at("ref-depth"));
}

void star::DynamicGeometryProcessor::drawRenderMaps(
    const unsigned frame_idx,
    cudaStream_t stream)
{
    // Generate new solver map (reference map inside)
    drawSolverMaps(
        frame_idx,
        m_buffer_idx,
        stream);
    // Generate new observation map (reference map inside)
    drawObservationMaps(
        frame_idx,
        m_buffer_idx,
        stream);
}

void star::DynamicGeometryProcessor::drawSolverMaps(
    const unsigned frame_idx,
    const unsigned geometry_idx,
    cudaStream_t stream)
{
    // Generate new reference map
    m_renderer->UnmapModelSurfelGeometryFromCuda(geometry_idx, stream);
    cudaSafeCall(cudaStreamSynchronize(stream));
    m_renderer->DrawSolverMapsWithRecentObservation(
        m_model_geometry[geometry_idx]->NumValidSurfels(),
        geometry_idx,
        0,
        frame_idx,
        m_cam2world.inverse());

    if (!m_solver_maps_mapped)
    {
        m_renderer->MapSolverMapsToCuda(m_solver_maps, stream);
        m_solver_maps_mapped = true;
    }
    m_renderer->MapModelSurfelGeometryToCuda(geometry_idx, stream);
    cudaSafeCall(cudaStreamSynchronize(stream));
}

void star::DynamicGeometryProcessor::drawObservationMaps(
    const unsigned frame_idx,
    const unsigned geometry_idx,
    cudaStream_t stream)
{
    // Generate new reference map
    m_renderer->UnmapModelSurfelGeometryFromCuda(geometry_idx, stream);
    cudaSafeCall(cudaStreamSynchronize(stream));
    m_renderer->DrawObservationMaps(
        m_model_geometry[geometry_idx]->NumValidSurfels(),
        geometry_idx, 0, frame_idx,
        m_cam2world.inverse(),
        true);

    if (!m_observation_maps_mapped)
    {
        m_renderer->MapObservationMapsToCuda(m_observation_maps, stream);
        m_observation_maps_mapped = true;
    }
    m_renderer->MapModelSurfelGeometryToCuda(geometry_idx, stream);
    cudaSafeCall(cudaStreamSynchronize(stream));
}