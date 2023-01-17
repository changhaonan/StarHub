#include <star/geometry/node_graph/Skinner.h>
#include <star/geometry/node_graph/NodeGraphManipulator.h>
#include <star/geometry/surfel/SurfelNodeDeformer.h>
#include <star/visualization/Visualizer.h>
#include <mono_star/common/ConfigParser.h>
#include "DynamicGeometryProcessor.h"

namespace star::device
{
    __global__ void GenerateSegmentationMapKernel(
        const ucharX<d_max_num_semantic> *__restrict__ semantic_prob,
        cudaTextureObject_t index_map,
        cudaSurfaceObject_t segmentation_map,
        const unsigned num_valid_surfel,
        const unsigned img_rows,
        const unsigned img_cols)
    {
        const auto idx = threadIdx.x + blockIdx.x * blockDim.x;
        const auto idy = threadIdx.y + blockIdx.y * blockDim.y;
        if (idx >= img_cols || idy >= img_rows)
            return;

        const auto surfel_idx = tex2D<int>(index_map, idx, idy);
        if (surfel_idx == 0xFFFFFFFF || surfel_idx >= num_valid_surfel)
            return;
        auto surfel_semantic = semantic_prob[surfel_idx];
        surf2Dwrite(max_id(surfel_semantic), segmentation_map, idx * sizeof(int), idy);
    }

}

star::DynamicGeometryProcessor::DynamicGeometryProcessor()
{
    std::cout << "Initilize DynamicGeometryProcessor..." << std::endl;
    auto &config = ConfigParser::Instance();

    m_data_geometry = std::make_shared<star::SurfelGeometry>();
    m_model_geometry[0] = std::make_shared<star::SurfelGeometry>();
    m_model_geometry[1] = std::make_shared<star::SurfelGeometry>();
    m_node_graph[0] = std::make_shared<star::NodeGraph>(config.node_radius());
    m_node_graph[1] = std::make_shared<star::NodeGraph>(config.node_radius());

    // Render
    m_renderer = std::make_shared<Renderer>(
        config.num_cam(),
        config.downsample_img_cols(),
        config.downsample_img_rows(),
        config.rgb_intrinsic_downsample(),
        config.max_rendering_depth());

    m_renderer->MapDataSurfelGeometryToCuda(0, *m_data_geometry);
    m_renderer->MapModelSurfelGeometryToCuda(0, *m_model_geometry[0]);
    m_renderer->MapModelSurfelGeometryToCuda(1, *m_model_geometry[1]);

    // Camera-related
    m_num_cam = config.num_cam();
    m_cam2world = config.extrinsic()[0];
    m_intrinsic = config.rgb_intrinsic_downsample();
    m_img_cols = config.downsample_img_cols();
    m_img_rows = config.downsample_img_rows();
    // Regulation
    m_dynamic_regulation = config.dynamic_regulation();
    // Other
    m_enable_semantic_surfel = config.enable_semantic_surfel();
    m_reinit_counter = config.reinit_counter();

    m_geometry_fusor = std::make_shared<GeometryFusor>(
        m_model_geometry,
        m_node_graph,
        m_renderer,
        m_num_cam,
        m_img_cols,
        m_img_rows,
        m_cam2world,
        m_intrinsic,
        m_enable_semantic_surfel,
        m_reinit_counter,
        m_dynamic_regulation);
    // Map
    createInt32TextureSurface(m_img_rows, m_img_cols, m_segmentation_map);

    // Vis
    m_enable_vis = config.enable_vis();
    m_pcd_size = config.pcd_size();
    m_node_graph_size = config.graph_node_size();

    // Eval
    m_track_semantic_label = config.track_semantic_label();
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

    releaseTextureCollect(m_segmentation_map);
}

void star::DynamicGeometryProcessor::ProcessFrame(
    const SurfelMapTex &surfel_map,
    const GArrayView<DualQuaternion> &solved_se3,
    const unsigned frame_idx,
    cudaStream_t stream)
{
    // Assign
    m_solved_se3 = solved_se3;

    // Update geometry
    if (frame_idx == 0)
    {
        initGeometry(surfel_map, m_cam2world, frame_idx, stream);
    }
    else if (frame_idx > 0)
    {
        updateGeometry(surfel_map, solved_se3, frame_idx, stream); // Apply warp
    }

    // Generate map from geometry
    drawRenderMaps(frame_idx, stream);
    computeSurfelMapTex();

    // Vis
    if (m_enable_vis)
        saveContext(frame_idx, stream);
}

void star::DynamicGeometryProcessor::initGeometry(
    const SurfelMapTex &surfel_map, const Eigen::Matrix4f &cam2world, const unsigned frame_idx, cudaStream_t stream)
{
    // Init Surfel geometry
    SurfelGeometryInitializer::InitFromGeometryMap(
        *m_model_geometry[m_buffer_idx],
        surfel_map,
        cam2world,
        m_enable_semantic_surfel,
        stream);

    // Init NodeGraph
    m_node_graph[m_buffer_idx]->InitializeNodeGraphFromVertex(
        m_model_geometry[m_buffer_idx]->LiveVertexConfidenceReadOnly(), frame_idx, false, stream);
    m_node_graph[m_buffer_idx]->ResetNodeGraphConnection(stream);

    // Perform Skinning without semantic
    auto geometry4skinner = m_model_geometry[m_buffer_idx]->GenerateGeometry4Skinner();
    auto node_graph4skinner = m_node_graph[m_buffer_idx]->GenerateNodeGraph4Skinner();
    Skinner::PerformSkinningFromLive(geometry4skinner, node_graph4skinner, stream);

    // Update with skinning with semantic
    if (m_enable_semantic_surfel)
    {
        // Update semantic prob
        NodeGraphManipulator::UpdateNodeSemanticProb(
            m_model_geometry[m_buffer_idx]->SurfelKNN().View(),
            m_model_geometry[m_buffer_idx]->SemanticProbReadOnly(),
            m_node_graph[m_buffer_idx]->GetNodeSemanticProb(),
            m_node_graph[m_buffer_idx]->GetNodeSemanticProbVoteBuffer(),
            stream);
        // Update node connection
        m_node_graph[m_buffer_idx]->ComputeNodeGraphConnectionFromSemantic(m_dynamic_regulation, stream);
        // Update surfel connection
        Skinner::UpdateSkinnningConnection(geometry4skinner, node_graph4skinner, stream);
    }
}

void star::DynamicGeometryProcessor::updateGeometry(
    const SurfelMapTex &surfel_map,
    const GArrayView<DualQuaternion> &solved_se3,
    const unsigned frame_idx,
    cudaStream_t stream)
{
    if (solved_se3.Size() == 0)
        return;

    // Apply the deformation
    SurfelNodeDeformer::ForwardWarpSurfelsAndNodes(
        m_node_graph[m_buffer_idx]->DeformAccess(), *m_model_geometry[m_buffer_idx], solved_se3, stream);

#ifdef ENABLE_POSE_EVAL
    // Update eval-related data
    m_node_graph[m_buffer_idx]->UpdateNodeDeformAcc(solved_se3, stream);
#endif

    // Init data geometry
    SurfelGeometryInitializer::InitFromGeometryMap(
        *m_data_geometry,
        surfel_map,
        m_cam2world,
        m_enable_semantic_surfel,
        stream);

    // Apply the geometry fusion
    m_geometry_fusor->Fuse(
        m_buffer_idx,
        frame_idx,
        surfel_map,
        m_data_geometry,
        stream);

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
    m_surfel_map_tex.color_time = 0;
    // Get observation-map
    m_surfel_map_tex.rgbd = m_observation_maps.rgbd_map[0];
    m_surfel_map_tex.depth = 0;
    m_surfel_map_tex.index = m_observation_maps.index_map[0];
    m_surfel_map_tex.segmentation = m_segmentation_map.texture;
    // Num
    m_surfel_map_tex.num_valid_surfel = m_model_geometry[m_buffer_idx]->NumValidSurfels();
}

void star::DynamicGeometryProcessor::saveContext(const unsigned frame_idx, cudaStream_t stream)
{
    auto &context = easy3d::Context::Instance();

    // Save Geometry
    unsigned last_buffer_idx = (m_buffer_idx + 1) % 2;
    unsigned vis_buffer_idx = (m_model_geometry[last_buffer_idx]->NumValidSurfels() > 0) ? last_buffer_idx : m_buffer_idx;
    // Exist valid last geo
    std::string ref_color_name = "ref_color";
    context.addPointCloud(ref_color_name, ref_color_name, m_cam2world.inverse(), m_pcd_size);
    visualize::SaveColoredPointCloud(
        m_model_geometry[vis_buffer_idx]->ReferenceVertexConfidenceReadOnly(),
        m_model_geometry[vis_buffer_idx]->ColorTimeReadOnly(),
        context.at(ref_color_name));

    std::string live_color_name = "live_color";
    context.addPointCloud(live_color_name, live_color_name, m_cam2world.inverse(), m_pcd_size);
    visualize::SaveColoredPointCloud(
        m_model_geometry[vis_buffer_idx]->LiveVertexConfidenceReadOnly(),
        m_model_geometry[vis_buffer_idx]->ColorTimeReadOnly(),
        context.at(live_color_name));

    // Save node graph
    context.addGraph("ref_graph", "", m_cam2world.inverse(), m_node_graph_size);
    visualize::SaveGraph(
        m_node_graph[vis_buffer_idx]->ReferenceNodeCoordinateReadOnly(),
        m_node_graph[vis_buffer_idx]->NodeKnnReadOnly(),
        context.at("ref_graph"));

    context.addGraph("live_graph", "", m_cam2world.inverse(), m_node_graph_size);
    visualize::SaveGraph(
        m_node_graph[vis_buffer_idx]->LiveNodeCoordinateReadOnly(),
        m_node_graph[vis_buffer_idx]->NodeKnnReadOnly(),
        context.at("live_graph"));

// #ifndef ENABLE_EFFICIENCY_MODE
    // Save semantic
    if (m_enable_semantic_surfel)
    {
        std::string semantic_pcd_name = "semantic_pcd";
        context.addPointCloud(semantic_pcd_name, semantic_pcd_name, m_cam2world.inverse(), m_pcd_size);
        visualize::SaveSemanticPointCloud(
            m_model_geometry[vis_buffer_idx]->LiveVertexConfidenceReadOnly(),
            m_model_geometry[vis_buffer_idx]->SemanticProbReadOnly(),
            visualize::default_semantic_color_dict,
            context.at(semantic_pcd_name));

        // Visualize for node graph
        std::string segmentation_graph_name = "segmentation_graph";
        context.addGraph(segmentation_graph_name, segmentation_graph_name, m_cam2world.inverse(), m_node_graph_size, 0.0, 1.0, true);

        // Transfer to color first
        std::vector<uchar3> node_vertex_color;
        visualize::Semantic2Color(
            m_node_graph[vis_buffer_idx]->GetNodeSemanticProbReadOnly(),
            visualize::default_semantic_color_dict,
            node_vertex_color);
        std::vector<float4> h_node_vertex;
        m_node_graph[vis_buffer_idx]->LiveNodeCoordinateReadOnly().Download(h_node_vertex);
        std::vector<ushortX<d_node_knn_size>> h_edges;
        m_node_graph[vis_buffer_idx]->NodeKnnReadOnly().Download(h_edges);
        std::vector<floatX<d_node_knn_size>> h_node_connect;
        m_node_graph[vis_buffer_idx]->NodeKnnConnectWeightReadOnly().Download(h_node_connect);
        visualize::SaveGraph(h_node_vertex, node_vertex_color, h_edges, h_node_connect, context.at(segmentation_graph_name));
    }
// #endif

    if (frame_idx > 0 && m_track_semantic_label > 0)
    {
        // Evaluate average dq when there is only one moving object
        // Select the first 6 nodes from semantic 3.
        unsigned num_node_selected = 6;
        DualQuaternion average_dq;
        float3 average_pos;
        computeAverageNodeDeform(vis_buffer_idx, m_track_semantic_label, num_node_selected, average_dq, average_pos, stream);
        auto average_mat = average_dq.se3_matrix();
        average_mat.trans = average_pos;
        context.addCoord("average_dq", "", m_cam2world.inverse() * Eigen::Matrix4f(average_mat), 0.3);
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
    // Generate new segmentation map
    drawSegmentationMap(
        frame_idx,
        m_buffer_idx,
        stream);
}

void star::DynamicGeometryProcessor::computeAverageNodeDeform(
    const unsigned buffer_idx,
    const unsigned short semantic_selected,
    const unsigned num_node_selected,
    DualQuaternion &average_node_deform,
    float3 &average_node_pos,
    cudaStream_t stream)
{
    // Fetch node list
    std::vector<unsigned short> node_list_selected;
    NodeGraphManipulator::SelectNodeBySemantic(
        m_node_graph[buffer_idx]->GetNodeSemanticProbReadOnly(),
        semantic_selected,
        num_node_selected,
        node_list_selected);

    // Compute average dq
    NodeGraphManipulator::AvergeNodeMovementAndPos(
        m_node_graph[buffer_idx]->LiveNodeCoordinateReadOnly(),
        m_solved_se3,
        node_list_selected,
        m_node_graph[buffer_idx]->NodeDeformAcc(),
        average_node_deform,
        average_node_pos);
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

void star::DynamicGeometryProcessor::drawSegmentationMap(
    const unsigned frame_idx, const unsigned geometry_idx, cudaStream_t stream)
{
    dim3 blk(32, 32);
    dim3 grid(divUp(m_img_rows, blk.x), divUp(m_img_cols, blk.y));
    device::GenerateSegmentationMapKernel<<<grid, blk, 0, stream>>>(
        m_model_geometry[geometry_idx]->SemanticProbReadOnly().Ptr(),
        m_observation_maps.index_map[0],
        m_segmentation_map.surface,
        m_model_geometry[geometry_idx]->NumValidSurfels(),
        m_img_rows,
        m_img_rows);
    cudaSafeCall(cudaStreamSynchronize(stream));
}