#pragma once

////Use materialized jtj?
//#define USE_MATERIALIZED_JTJ
//
////Use dense solver maps
//#define USE_DENSE_SOLVER_MAPS
//
////Use huber weights in sparse feature term to deal with outlier
//#define USE_FEATURE_HUBER_WEIGHT
//
////If the node graph are too far-awary from each other, they are unlikely valid
////#define CLIP_FARAWAY_NODEGRAPH_PAIR  // Add this for locate floating
//
////Enable group graph
//#define ENABLE_GROUP_GRAPH

////Clip the image term if they are too large
////#define USE_IMAGE_HUBER_WEIGHT
//constexpr float d_density_map_cutoff = 0.3f;
//
//The value of invalid index map
//
//The device accessed constant for find corresponded depth pairs
constexpr float d_correspondence_normal_dot_threshold  = 0.7f;
constexpr float d_correspondence_distance_threshold = 0.03f;
constexpr float d_correspondence_distance_threshold_square = (d_correspondence_distance_threshold * d_correspondence_distance_threshold);

////The upperbound of alignment error
//constexpr float d_maximum_alignment_error = 0.005f;  // 5mm, for multi-surface, thin board
//
////The weight between different terms, these are deprecated
//constexpr float lambda_smooth = 2.0f;
//constexpr float lambda_smooth_square = (lambda_smooth * lambda_smooth);
//
//// Precise error check
//constexpr float d_node_outlier_error_threshold = 0.01f;