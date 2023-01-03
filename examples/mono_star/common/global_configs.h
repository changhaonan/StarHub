#pragma once

// Disable cuda on Eigen
#ifndef EIGEN_NO_CUDA
#define EIGEN_NO_CUDA
#endif

// Perform debug sync and check cuda error
//#define CUDA_DEBUG_SYNC_CHECK

// For pcl access of new in debug mode
#if defined(CUDA_DEBUG_SYNC_CHECK)
#define EIGEN_DONT_VECTORIZE
#define EIGEN_DISABLE_UNALIGNED_ARRAY_ASSERT
#endif

#define CUDA_CHECKERR_SYNC 1

// Geometry-related
#include <star/geometry/constants.h>

// Opt-related
#include <star/opt/constants.h>

// The scale of fusion map, will be accessed on device
#define d_fusion_map_scale 4

// The scale of filter map, will be accessed on device
#define d_filter_map_scale 4 // The same as d_fusion_map_scale

// Normalize the interpolate weight to 1
#define USE_INTERPOLATE_WEIGHT_NORMALIZATION

// Fix boost broken issue with cuda compile
#ifdef __CUDACC__
#define BOOST_PP_VARIADICS 0
#endif

// Camera-related
constexpr unsigned d_max_cam = 1;

// NVTX trace
//#define OPTIMIZE

// FPS setting
constexpr unsigned d_fps = 10;

// IO debug
#define USE_IO_DEBUG

// Debug
//#define CUDA_DEBUG_SYNC_CHECK
constexpr unsigned d_bin_size = 32;      // Bin size is fixed to 32 to match warp size
constexpr unsigned d_max_num_bin = 1024; // Fixed as 32 * 32

constexpr unsigned d_invalid_index = 0xFFFFFFFF;
constexpr float d_correspondence_normal_dot_threshold = 0.7f;
constexpr float d_correspondence_distance_threshold = 0.03f;
constexpr float d_correspondence_distance_threshold_square = (d_correspondence_distance_threshold * d_correspondence_distance_threshold);

// WarpLevel
constexpr unsigned preconditioner_blk_size = d_node_variable_dim_square;
// Warpsize is 32, what we defined is thread we used in each block, has to be 2^n
constexpr unsigned opt_warp_size = 32;

// Maximum number of semantic types
constexpr unsigned d_max_num_semantic = 19;

// Opt debug
// #define OPT_DEBUG_CHECK

// Geometry debug
#define DYNAMIC_GEOMETRY_DEBUG

// Eval
#define ENABLE_POSE_EVAL