#include <star/geometry/node_graph/Skinner.h>
#include <star/geometry/node_graph/NodeGraphManipulator.h>
#include <star/geometry/surfel/SurfelNodeDeformer.h>
#include <star/visualization/Visualizer.h>
#include <mono_star/common/ConfigParser.h>
#include "DynamicGeometryProcessor.h"

star::DynamicGeometryProcessor::DynamicGeometryProcessor()
{
    std::cout << "Initilize DynamicGeometryProcessor..." << std::endl;
    auto &config = ConfigParser::Instance();

    m_data_geometry = std::make_shared<star::SurfelGeometry>();
    m_model_geometry[0] = std::make_shared<star::SurfelGeometry>();
    m_model_geometry[1] = std::make_shared<star::SurfelGeometry>();
    m_model_keypoints[0] = std::make_shared<star::KeyPoints>(config.keypoint_type());
    m_model_keypoints[1] = std::make_shared<star::KeyPoints>(config.keypoint_type());
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
    const float kp_match_threshold = 0.1;
    m_keypoint_fusor = std::make_shared<KeyPointFusor>(kp_match_threshold);

    // Vis
    m_enable_vis = config.enable_vis();
    m_pcd_size = config.pcd_size();
    m_node_graph_size = config.graph_node_size();
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
    const SurfelMapTex &surfel_map,
    const GArrayView<float2> &keypoints,
    const GArrayView<float> &descriptors,
    const GArrayView<int2> &kp_matches,
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
        initKeyPoints(surfel_map, keypoints, descriptors, m_cam2world, frame_idx, stream);
    }
    else if (frame_idx > 0)
    {
        updateGeometry(surfel_map, keypoints, descriptors, kp_matches, solved_se3, frame_idx, stream); // Apply warp
        // Update the keypoints 10 frames
        // TODO: test this in the future
        // if (frame_idx % 10 == 0)
        //     initKeyPoints(surfel_map, keypoints, descriptors, m_cam2world, frame_idx, stream);
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

void star::DynamicGeometryProcessor::initKeyPoints(
    const SurfelMapTex &surfel_map,
    const GArrayView<float2> &keypoints,
    const GArrayView<float> &descriptors,
    const Eigen::Matrix4f &cam2world,
    const unsigned frame_idx,
    cudaStream_t stream)
{
    initKeyPointsGeometry(surfel_map, keypoints, descriptors, cam2world, m_buffer_idx, stream);
    updateKeyPointSkinning(stream);
}

void star::DynamicGeometryProcessor::initKeyPointsGeometry(
    const SurfelMapTex &surfel_map,
    const GArrayView<float2> &keypoints,
    const GArrayView<float> &descriptors,
    const Eigen::Matrix4f &cam2world,
    const unsigned idle_buffer_idx,
    cudaStream_t stream)
{
    // Resize keypoints
    m_model_keypoints[idle_buffer_idx]->Resize(keypoints.Size());
    // Init keypoint geometry
    SurfelGeometryInitializer::InitFromGeometryMap(
        *m_model_keypoints[idle_buffer_idx],
        surfel_map,
        keypoints,
        cam2world,
        m_enable_semantic_surfel,
        stream);
    // Init keypoint descriptor
    cudaSafeCall(cudaMemcpyAsync(
        m_model_keypoints[idle_buffer_idx]->Descriptor().Ptr(),
        descriptors.Ptr(),
        descriptors.ByteSize(),
        cudaMemcpyDeviceToDevice,
        stream));
}

void star::DynamicGeometryProcessor::updateKeyPointSkinning(
    cudaStream_t stream)
{
    // Perform Skinning without semantic
    auto geometry4skinner = m_model_keypoints[m_buffer_idx]->GenerateGeometry4Skinner();
    auto node_graph4skinner = m_node_graph[m_buffer_idx]->GenerateNodeGraph4Skinner();
    Skinner::PerformSkinningFromLive(geometry4skinner, node_graph4skinner, stream);

    // Update with skinning with semantic
    if (m_enable_semantic_surfel)
    {
        // Update surfel connection
        Skinner::UpdateSkinnningConnection(geometry4skinner, node_graph4skinner, stream);
    }
}

void star::DynamicGeometryProcessor::updateGeometry(
    const SurfelMapTex &surfel_map,
    const GArrayView<float2> &keypoints,
    const GArrayView<float> &descriptors,
    const GArrayView<int2> &kp_matches,
    const GArrayView<DualQuaternion> &solved_se3,
    const unsigned frame_idx,
    cudaStream_t stream)
{
    if (solved_se3.Size() == 0)
        return;

    // Apply the deformation
    SurfelNodeDeformer::ForwardWarpSurfelsAndNodes(
        m_node_graph[m_buffer_idx]->DeformAccess(), *m_model_geometry[m_buffer_idx], solved_se3, stream);
    SurfelNodeDeformer::ForwardWarpSurfelsAndNodes(
        m_node_graph[m_buffer_idx]->DeformAccess(), *m_model_keypoints[m_buffer_idx], solved_se3, stream);

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

#define RESET_KEYPOINTS
#ifndef RESET_KEYPOINTS
    // Apply the keypoint fusion
    const auto idle_buffer_idx = (m_buffer_idx + 1) & 1;
    initKeyPointsGeometry(surfel_map, keypoints, descriptors, m_cam2world, idle_buffer_idx, stream);
    m_keypoint_fusor->Fuse(
        m_model_keypoints[m_buffer_idx],
        m_model_keypoints[idle_buffer_idx],
        kp_matches,
        (frame_idx % 10 == 0),
        stream);
    updateKeyPointSkinning(stream);
#else
    // Reset keypoints
    if (frame_idx % 10 == 0)
    {
        initKeyPoints(surfel_map, keypoints, descriptors, m_cam2world, frame_idx, stream);
    }
#endif

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
    // Reanchor the keypoints
    KeyPoints::ReAnchor(
        m_model_keypoints[m_buffer_idx],
        m_model_keypoints[next_buffer_idx],
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

#ifndef ENABLE_EFFICIENCY_MODE
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
#endif

#ifdef ENABLE_POSE_EVAL
    if (frame_idx > 0)
    {
        // Evaluate average dq when there is only one moving object
        // Select the first 6 nodes from semantic 3.
        unsigned short semantic_id = 3;
        unsigned num_node_selected = 6;
        DualQuaternion average_dq;
        float3 average_pos;
        computeAverageNodeDeform(vis_buffer_idx, semantic_id, num_node_selected, average_dq, average_pos, stream);
        auto average_mat = average_dq.se3_matrix();
        average_mat.trans = average_pos;
        context.addCoord("average_dq", "", m_cam2world.inverse() * Eigen::Matrix4f(average_mat), 0.3);
    }
#endif
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