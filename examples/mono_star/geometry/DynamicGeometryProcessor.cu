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

    // Camera-related
    m_cam2world = config.extrinsic()[0];
}

star::DynamicGeometryProcessor::~DynamicGeometryProcessor()
{
    m_renderer->UnmapDataSurfelGeometryFromCuda(0);
    m_renderer->UnmapModelSurfelGeometryFromCuda(0);
    m_renderer->UnmapModelSurfelGeometryFromCuda(1);
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
}

void star::DynamicGeometryProcessor::initGeometry(
    const SurfelMap &surfel_map, const Eigen::Matrix4f &cam2world, const unsigned frame_idx, cudaStream_t stream)
{
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

void star::DynamicGeometryProcessor::saveContext(const unsigned frame_idx, cudaStream_t stream) {
    auto& context = easy3d::Context::Instance();

    // Save node graph
    context.addGraph("live_graph", "", m_cam2world.inverse());
    visualize::SaveGraph(
        m_node_graph[m_buffer_idx]->GetLiveNodeCoordinate(),
        m_node_graph[m_buffer_idx]->GetNodeKnn(),
        context.at("live_graph")
    );
}