#pragma once
#include <star/common/common_types.h>
#include <star/common/common_texture_utils.h>
#include <star/common/ArrayView.h>
#include <star/common/ArraySlice.h>
#include <star/common/constants.h>
#include <star/geometry/constants.h>

namespace star
{

    struct FusionMaps
    {
        using Ptr = std::shared_ptr<FusionMaps>;
        cudaTextureObject_t warp_vertex_confid_map[d_max_cam] = {0};
        cudaTextureObject_t warp_normal_radius_map[d_max_cam] = {0};
        cudaTextureObject_t index_map[d_max_cam] = {0};
        cudaTextureObject_t color_time_map[d_max_cam] = {0};
    };

    struct Geometry4Fusion
    {
        GArraySlice<float4> vertex_confid;
        GArraySlice<float4> normal_radius;
        GArraySlice<float4> color_time;
        unsigned num_valid_surfel = 0;
    };

    struct Measure4Fusion
    {
        cudaTextureObject_t vertex_confid_map[d_max_cam] = {0};
        cudaTextureObject_t normal_radius_map[d_max_cam] = {0};
        cudaTextureObject_t color_time_map[d_max_cam] = {0};
        cudaTextureObject_t index_map[d_max_cam] = {0};
        unsigned num_valid_surfel = 0;
    };

    /* Only used for geometry removal, so don't need too much
     */
    struct Measure4GeometryRemoval
    {
        cudaTextureObject_t depth4removal_map[d_max_cam] = {0};
    };

    struct Geometry4SemanticFusion
    {
        GArraySlice<ucharX<d_max_num_semantic>> semantic_prob;
        unsigned num_valid_surfel = 0;
    };

    struct Segmentation4SemanticFusion
    {
        cudaTextureObject_t segmentation[d_max_cam] = {0};
    };

    /* For Geometry Add
     */
    struct GeometryCandidateIndicator
    {
        GArraySlice<unsigned> candidate_validity_indicator;
        GArraySlice<unsigned> candidate_unsupported_indicator;
    };

    // Geometry Add, but additional info
    struct GeometryCandidatePlus
    {
        GArrayView<unsigned> candidate_validity_indicator;
        GArrayView<unsigned> candidate_validity_indicator_prefixsum;
        GArrayView<ushortX<d_surfel_knn_size>> append_candidate_surfel_knn;
        unsigned num_valid_candidate = 0;
        unsigned num_supported_candidate = 0;
    };

    struct Geometry4GeometryAppend
    {
        // Candidate
        GArrayView<float4> vertex_confid_append_candidate;
        GArrayView<float4> normal_radius_append_candidate;
        GArrayView<float4> color_time_append_candidate;
        GArrayView<ucharX<d_max_num_semantic>> semantic_prob_append_candidate; // (Optional) one
        unsigned num_append_candidate = 0;
    };

    struct Geometry4GeometryRemaining
    {
        GArrayView<unsigned> remaining_indicator;
        GArrayView<unsigned> remaining_indicator_prefixsum;
        unsigned num_remaining_surfel = 0;
    };

    /** \brief Observation is different from measurement.
     * Measure is measured from environment.
     * Observation comes from geometry
     */
    struct ObservationMaps
    {
        using Ptr = std::shared_ptr<ObservationMaps>;
        cudaTextureObject_t rgbd_map[d_max_cam];
        cudaTextureObject_t index_map[d_max_cam];
    };

    /* Input for solver, Measurement, 2D, image-level
     */
    struct Measure4Solver
    {
        cudaTextureObject_t vertex_confid_map[d_max_cam];
        cudaTextureObject_t normal_radius_map[d_max_cam];
        cudaTextureObject_t index_map[d_max_cam];
        unsigned num_cam;
    };

    /* Input for solver, Render, 2D, image-level
     */
    struct Render4Solver
    {
        cudaTextureObject_t reference_vertex_map[d_max_cam];
        cudaTextureObject_t reference_normal_map[d_max_cam];
        cudaTextureObject_t index_map[d_max_cam];
        unsigned num_cam;
    };

    // Solver map
    struct SolverMaps
    {
        using Ptr = std::shared_ptr<SolverMaps>;

        cudaTextureObject_t reference_vertex_map[d_max_cam] = {0};
        cudaTextureObject_t reference_normal_map[d_max_cam] = {0};
        cudaTextureObject_t index_map[d_max_cam] = {0};
        cudaTextureObject_t normalized_rgbd_map[d_max_cam] = {0};
    };

    /* Input for solver, Geometry, 1D, surfel-level
     */
    struct Geometry4Solver
    {
        GArrayView<ushortX<d_surfel_knn_size>> surfel_knn;
        GArrayView<floatX<d_surfel_knn_size>> surfel_knn_spatial_weight;
        GArrayView<floatX<d_surfel_knn_size>> surfel_knn_connect_weight;
        unsigned num_vertex;
    };

    /* Input for solver, NodeGraph, 1D, node-level
     */
    struct NodeGraph4Solver
    {
        // Used for reg term
        GArrayView<float4> reference_node_coords;
        GArrayView<ushort3> node_graph;
        GArrayView<floatX<d_node_knn_size>> node_knn_connect_weight;
        // Used for node motion term
        GArrayView<ushortX<d_surfel_knn_size>> nodel_knn;
        GArrayView<floatX<d_surfel_knn_size>> node_knn_spatial_weight;
        unsigned num_node;
        float node_radius_square;
    };

    /* Input for solver, NodeFlow, 1D, node-level
     */
    struct NodeFlow4Solver
    {
        GArrayView<float4> node_motion_pred;
        unsigned num_node;
    };

    /* Input for solver, OpticalFlow, 2D, image-level
     */
    struct OpticalFlow4Solver
    {
        cudaTextureObject_t opticalflow_map[d_max_cam];
        unsigned num_cam;
    };
}