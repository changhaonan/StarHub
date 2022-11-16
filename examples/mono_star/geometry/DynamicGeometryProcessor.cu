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

void star::DynamicGeometryProcessor::Process(
    StarStageBuffer &star_stage_buffer_this,
    const StarStageBuffer &star_stage_buffer_prev,
    cudaStream_t stream,
    const unsigned frame_idx)
{
}

void star::DynamicGeometryProcessor::processFrame(
    const unsigned frame_idx,
    cudaStream_t stream)
{
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
    m_buffer_idx = (m_buffer_idx + 1) % 2;

    // Init Surfel geometry
    SurfelGeometryInitializer::InitFromGeometryMap(
        *m_model_geometry[m_buffer_idx],
        surfel_map,
        cam2world,
        stream);

    // Init NodeGraph
    m_node_graph[m_buffer_idx]->InitializeNodeGraphFromVertex(
        m_model_geometry[m_buffer_idx]->LiveVertexConfidenceReadOnly(), frame_idx, false, stream);
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
    if (m_model_geometry[last_buffer_idx]->NumValidSurfels() > 0) {  // Exist valid last geo
        context.addPointCloud(last_geo_name, last_geo_name, m_cam2world.inverse());
        visualize::SavePointCloud(
            m_model_geometry[last_buffer_idx]->ReferenceVertexConfidenceReadOnly(),
            context.at(last_geo_name));
    }

    // Save node graph
    context.addGraph("live_graph", "", m_cam2world.inverse());
    visualize::SaveGraph(
        m_node_graph[m_buffer_idx]->GetLiveNodeCoordinate(),
        m_node_graph[m_buffer_idx]->GetNodeKnn(),
        context.at("live_graph"));

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