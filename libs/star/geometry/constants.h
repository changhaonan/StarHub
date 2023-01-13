#pragma once

namespace star
{
    // Structure
    constexpr unsigned d_node_knn_size = 4;
    constexpr unsigned d_surfel_knn_size = d_node_knn_size;
    constexpr unsigned d_surfel_knn_pair_size = d_surfel_knn_size * (d_surfel_knn_size - 1) / 2;

    // Buffer size
    constexpr unsigned d_max_num_surfels = 800000;
    constexpr unsigned d_max_num_surfel_candidates = 200000;
    constexpr unsigned d_max_num_semantic = 20;
    constexpr unsigned d_max_num_nodes = 2048;

    // Render related
    constexpr int d_stable_surfel_confidence_threshold = 1;
    constexpr int d_rendering_recent_time_threshold = 3;
    constexpr int d_fusion_map_scale = 4;

    // Keypoint related
    constexpr unsigned d_max_num_keypoints = 5000;
}