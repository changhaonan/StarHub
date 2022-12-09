#include "DynamicGeometryFusor.h"
#include <star/geometry/node_graph/NodeGraphManipulator.h>
#include <star/geometry/node_graph/Skinner.h>

void star::DynamicGeometryFusor::drawFusionMaps(
    const unsigned frame_idx,
    const unsigned geometry_idx,
    cudaStream_t stream)
{
    // Generate new reference map
    m_renderer->UnmapModelSurfelGeometryFromCuda(geometry_idx, stream);
    cudaSafeCall(cudaStreamSynchronize(stream));
    for (auto cam_idx = 0; cam_idx < m_num_cam; ++cam_idx)
    {
        m_renderer->DrawFusionMaps(
            m_model_surfel_geometry[geometry_idx]->NumValidSurfels(),
            geometry_idx,
            cam_idx,
            m_cam2world[cam_idx].inverse());
    }
    if (!m_fusion_maps_mapped)
    {
        m_renderer->MapFusionMapsToCuda(m_fusion_maps, stream);
        m_fusion_maps_mapped = true;
    }
    m_renderer->MapModelSurfelGeometryToCuda(geometry_idx, stream);
    cudaSafeCall(cudaStreamSynchronize(stream));
}

void star::DynamicGeometryFusor::geometryRemoval(
    const Measure4Fusion &measure4fusion,
    const unsigned current_time,
    unsigned &current_geometry_idx,
    unsigned &current_node_graph_idx,
    cudaStream_t stream)
{
    // Draw fusion map first
    drawFusionMaps(
        current_time,
        current_geometry_idx,
        stream);

    // Set remaining marker
    m_fusion_remaining_surfel_marker->Initialization(
        m_model_surfel_geometry[current_geometry_idx]->NumValidSurfels(), stream);
    m_fusion_remaining_surfel_marker->SetInputs(
        measure4fusion,
        m_fusion_maps,
        float(current_time),
        m_intrinsic,
        m_cam2world);
    m_fusion_remaining_surfel_marker->UpdateRemainingSurfelIndicator(stream);
    m_fusion_remaining_surfel_marker->PostProcessRemainingSurfelIndicator(stream);
    m_fusion_remaining_surfel_marker->RemainingSurfelIndicatorPrefixSumSync(stream);
    Geometry4GeometryAppend empty_append;
    GeometryCandidatePlus empty_candidate_plus;
    Geometry4GeometryRemaining geometry4geometry_remaining = m_fusion_remaining_surfel_marker->GenerateGeometry4GeometryRemaining();

    // Remove Node graph by setting NodeStatus
    NodeGraphManipulator::UpdateCounterNodeOutTrack(
        geometry4geometry_remaining.remaining_indicator,
        m_model_surfel_geometry[current_geometry_idx]->SurfelKNNReadOnly(),
        m_node_graph[current_node_graph_idx]->GetCounterNodeOuttrack(),
        stream);
    unsigned num_node_remove_count;
    NodeGraphManipulator::RemoveNodeOutTrackSync(
        m_node_graph[current_node_graph_idx]->GetNodeKnn(),
        m_node_graph[current_node_graph_idx]->GetCounterNodeOuttrackReadOnly(),
        m_node_graph[current_node_graph_idx]->GetNodeStatus(),
        m_node_graph[current_node_graph_idx]->GetNodeDistance(),
        num_node_remove_count,
        m_counter_node_outtrack_threshold,
        m_frozen_time,
        stream);
    // Compact
    auto compact_buffer_idx = (current_geometry_idx + 1) % 2; // Type is important!!!
    m_geometry_compact_handler->SetInputs(
        m_model_surfel_geometry[current_geometry_idx],
        m_model_surfel_geometry[compact_buffer_idx],
        empty_append,
        empty_candidate_plus,
        geometry4geometry_remaining);
    m_geometry_compact_handler->CompactLiveSurfelToAnotherBufferRemainingOnlySync(m_use_semantic, stream);

    // Log
    auto num_prev_surfel_size = m_model_surfel_geometry[current_geometry_idx]->NumValidSurfels();
    auto num_removed_surfel = num_prev_surfel_size - m_fusion_remaining_surfel_marker->GetRemainingSurfelSize();
    std::cout << "[Info]: Surfel Removed by " << num_removed_surfel << "; Node remove by " << num_node_remove_count << "." << std::endl;

    // Update idx
    current_geometry_idx = compact_buffer_idx;
}

void star::DynamicGeometryFusor::geometryRemovalSurfelWarp(
    const Measure4GeometryRemoval &meaure4geometry_removal,
    const unsigned current_time,
    unsigned &current_geometry_idx,
    unsigned &current_node_graph_idx,
    cudaStream_t stream)
{
    // 1. Prepare input
    // 1.1. Fusion map
    drawFusionMaps(
        current_time,
        current_geometry_idx,
        stream);
    // 1.2. Gometry4fusion
    auto geometry4fusion = m_model_surfel_geometry[current_geometry_idx]->GenerateGeometry4Fusion(false);
    // 1.3. Set remaining marker
    m_fusion_remaining_surfel_marker->InitializationSurfelWarp(
        m_model_surfel_geometry[current_geometry_idx]->NumValidSurfels(), stream);
    m_fusion_remaining_surfel_marker->SetInputs(
        meaure4geometry_removal,
        geometry4fusion,
        m_fusion_maps,
        float(current_time),
        m_intrinsic,
        m_cam2world);

    // 2. Compute remaining
    m_fusion_remaining_surfel_marker->UpdateRemainingSurfelIndicatorSurfelWarp(stream);
    m_fusion_remaining_surfel_marker->RemainingSurfelIndicatorPrefixSumSync(stream);
    // std::cout << "Remaining: " << m_fusion_remaining_surfel_marker->GetRemainingSurfelSize() << std::endl;
    Geometry4GeometryAppend empty_append;
    GeometryCandidatePlus empty_candidate_plus;
    Geometry4GeometryRemaining geometry4geometry_remaining = m_fusion_remaining_surfel_marker->GenerateGeometry4GeometryRemaining();

    // 3. Remove Node graph by setting NodeStatus
    // NodeGraphManipulator::UpdateCounterNodeOutTrack(
    //	geometry4geometry_remaining.remaining_indicator,
    //	m_model_surfel_geometry[current_geometry_idx]->SurfelKNNReadOnly(),
    //	m_node_graph[current_node_graph_idx]->GetCounterNodeOuttrack(),
    //	stream
    //);
    // unsigned num_node_remove_count;
    // NodeGraphManipulator::RemoveNodeOutTrackSync(
    //	m_node_graph[current_node_graph_idx]->GetNodeKnn(),
    //	m_node_graph[current_node_graph_idx]->GetCounterNodeOuttrackReadOnly(),
    //	m_node_graph[current_node_graph_idx]->GetNodeStatus(),
    //	m_node_graph[current_node_graph_idx]->GetNodeDistance(),
    //	num_node_remove_count,
    //	m_counter_node_outtrack_threshold,
    //	m_frozen_time,
    //	stream
    //);
    // 4. Compact
    std::cout << "Before: " << m_model_surfel_geometry[current_geometry_idx]->NumValidSurfels() << std::endl;
    auto compact_buffer_idx = (current_geometry_idx + 1) % 2; // Type is important!!!
    m_geometry_compact_handler->SetInputs(
        m_model_surfel_geometry[current_geometry_idx],
        m_model_surfel_geometry[compact_buffer_idx],
        empty_append,
        empty_candidate_plus,
        geometry4geometry_remaining);
    m_geometry_compact_handler->CompactLiveSurfelToAnotherBufferRemainingOnlySync(m_use_semantic, stream);
    std::cout << "After: " << m_model_surfel_geometry[current_geometry_idx]->NumValidSurfels() << std::endl;

    // 5. Log
    auto num_prev_surfel_size = m_model_surfel_geometry[current_geometry_idx]->NumValidSurfels();
    auto num_removed_surfel = num_prev_surfel_size - m_fusion_remaining_surfel_marker->GetRemainingSurfelSize();
    std::cout << "[Info]: Surfel Removed by " << num_removed_surfel << ", remaining "
              << geometry4geometry_remaining.num_remaining_surfel
              << ", prev: " << num_prev_surfel_size << std::endl;
    //	//<< "; Node remove by " << num_node_remove_count << "." << std::endl;

    // 6. Update idx
    current_geometry_idx = compact_buffer_idx;
}

void star::DynamicGeometryFusor::geometryFusion(
    const Measure4Fusion &measure4fusion,
    const Segmentation4SemanticFusion &segmentation4semantic_fusion,
    Geometry4GeometryAppend &geometry4geometry_append,
    const unsigned current_time,
    unsigned &current_geometry_idx,
    const bool update_semantic,
    const bool geometry_reinit,
    cudaStream_t stream)
{
    // 1. Draw fusion map first
    drawFusionMaps(
        current_time,
        current_geometry_idx,
        stream);

    // 2. Set input
    auto geometry4fusion = m_model_surfel_geometry[current_geometry_idx]->GenerateGeometry4Fusion(false); // Fusion is an inplace operation
    m_surfel_fusion_handler->ZeroInitializeIndicator(
        geometry4fusion.num_valid_surfel,
        measure4fusion.num_valid_surfel,
        stream);
    if (!update_semantic)
    {
        m_surfel_fusion_handler->SetInputs(
            m_fusion_maps,
            measure4fusion,
            geometry4fusion,
            current_time,
            m_cam2world);
        std::cout << "Start fusion without semantic" << std::endl;
    }
    else
    {
        auto geometry4semantic_fusion = m_model_surfel_geometry[current_geometry_idx]->GenerateGeometry4SemanticFusion();
        m_surfel_fusion_handler->SetInputs(
            m_fusion_maps,
            measure4fusion,
            segmentation4semantic_fusion,
            geometry4fusion,
            geometry4semantic_fusion,
            current_time,
            m_cam2world);
        std::cout << "Start fusion with semantic" << std::endl;
    }

    // 3. Process fusion
    if (!geometry_reinit)
    {
        m_surfel_fusion_handler->ProcessFusion(update_semantic, stream);
    }
    else
    {
        // m_surfel_fusion_handler->ProcessFusionReinit(stream);
        m_surfel_fusion_handler->ProcessFusion(update_semantic, stream);
    }

    // 4. Get append candidate
    if (!update_semantic)
    {
        m_surfel_fusion_handler->CompactAppendedCandidate(
            m_data_surfel_geometry->LiveVertexConfidenceReadOnly(),
            m_data_surfel_geometry->LiveNormalRadiusReadOnly(),
            m_data_surfel_geometry->ColorTimeReadOnly(),
            stream);
    }
    else
    {
        m_surfel_fusion_handler->CompactAppendedCandidate(
            m_data_surfel_geometry->LiveVertexConfidenceReadOnly(),
            m_data_surfel_geometry->LiveNormalRadiusReadOnly(),
            m_data_surfel_geometry->ColorTimeReadOnly(),
            m_data_surfel_geometry->SemanticProbReadOnly(),
            stream);
    }

    // 5. Write the candidate to append & geometry_idx
    geometry4geometry_append = m_surfel_fusion_handler->GenerateGeometry4GeometryAppend();
}

void star::DynamicGeometryFusor::geometryAppend(
    const Geometry4GeometryAppend &geometry4geometry_append,
    const unsigned current_time,
    unsigned &current_geometry_idx,
    unsigned &current_node_graph_idx,
    cudaStream_t stream)
{
    m_geometry_append_handler->SetInputs(geometry4geometry_append.vertex_confid_append_candidate);
    m_geometry_append_handler->Initialize(stream);
    auto geometry_candidate_indicator = m_geometry_append_handler->GenerateGeometryCandidateIndicator();
    // 1. Compute support node
    NodeGraphManipulator::CheckSurfelCandidateSupportStatus(
        geometry4geometry_append.vertex_confid_append_candidate,
        m_node_graph[current_node_graph_idx]->GetLiveNodeCoordinate(),
        m_node_graph[current_node_graph_idx]->GetNodeStatusReadOnly(),
        geometry_candidate_indicator.candidate_validity_indicator,
        geometry_candidate_indicator.candidate_unsupported_indicator,
        m_geometry_append_handler->AppendSurfelKnn(),
        stream,
        m_node_graph[current_node_graph_idx]->GetNodeRadiusSquare());
    m_geometry_append_handler->PrefixSumSync(stream); // Sum
    auto geometry_candidate_plus = m_geometry_append_handler->GenerateGeometryCandidatePlus();
    // 2. Append node graph [Todo]
    // 2.1. Get unsupported surfel
    m_geometry_append_handler->ComputeUnsupportedCandidate(stream);
    auto unsupported_candidate_surfel = m_geometry_append_handler->GenerateUnsupportedCandidate();
    std::cout << "Unsupported surfel num is " << unsupported_candidate_surfel.Size() << std::endl;
    // 2.2. Extract node form unsupported surfel & Append node to node graph & Update node graph
    m_node_graph[current_node_graph_idx]->NaiveExpandNodeGraphFromVertexUnsupported(
        unsupported_candidate_surfel,
        current_time,
        stream);

    // 3. Compact geometry
    Geometry4GeometryRemaining empty_remaining;
    auto compact_buffer_idx = (current_geometry_idx + 1) % 2; // Type is important!!!
    m_geometry_compact_handler->SetInputs(
        m_model_surfel_geometry[current_geometry_idx],
        m_model_surfel_geometry[compact_buffer_idx],
        geometry4geometry_append,
        geometry_candidate_plus,
        empty_remaining);
    m_geometry_compact_handler->CompactLiveSurfelToAnotherBufferAppendOnlySync(m_use_semantic, stream);

    // 4. Update idx
    current_geometry_idx = compact_buffer_idx;
}

void star::DynamicGeometryFusor::geometrySkinning(
    const unsigned num_remaining_surfel,
    const bool is_incremental,
    unsigned &current_geometry_idx,
    unsigned &current_node_graph_idx,
    cudaStream_t stream)
{
    if (is_incremental && m_use_semantic)
    {
        auto geometry_skinner = m_model_surfel_geometry[current_geometry_idx]->GenerateGeometry4Skinner();
        auto node_graph_skinner = m_node_graph[current_node_graph_idx]->GenerateNodeGraph4Skinner();

        // 1. Update the surfel skinning (Skinner)
        Skinner::PerformIncSkinnningFromLive(
            geometry_skinner,
            node_graph_skinner,
            m_node_graph[current_node_graph_idx]->GetPrevNodeSize(),
            num_remaining_surfel,
            stream);

        // 2. Update the newly added node semantic (NodeGraphManipulator)
        NodeGraphManipulator::UpdateIncNodeSemanticProb(
            m_model_surfel_geometry[current_geometry_idx]->SurfelKNN().View(),
            m_model_surfel_geometry[current_geometry_idx]->SemanticProbReadOnly(),
            m_node_graph[current_node_graph_idx]->GetNodeSemanticProb(),
            m_node_graph[current_node_graph_idx]->GetNodeSemanticProbVoteBuffer(),
            m_node_graph[current_node_graph_idx]->GetPrevNodeSize(),
            stream);

        // 3. Update the newly added node connection (NodeGraph)
        const bool building_use_ref = false;
        m_node_graph[current_node_graph_idx]->BuildNodeGraphFromScratch(building_use_ref, stream);
        m_node_graph[current_node_graph_idx]->ComputeNodeGraphConnectionFromSemantic(m_dynamic_regulation, stream);

        // 4. Update the surfel skinning connection (Skinner)
        Skinner::UpdateSkinnningConnection(
            geometry_skinner,
            node_graph_skinner,
            stream);
    }
    else
    {
        printf("Geometry skinning is not implemented for this setting!\n");
    }
}
