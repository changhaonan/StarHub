#pragma once

namespace star
{
    constexpr float d_tsdf_voxel_size = 0.005f;
    constexpr float d_tsdf_threshold = 3.0f * d_tsdf_voxel_size;
    constexpr float d_near_clip = 0.3f;
    constexpr float d_far_clip = 10.f;
    constexpr float d_tsdf_weight_max = 128.f;
    // Interpolation
    constexpr float d_interpolation_gap = 0.02f; // When faced with gap between pixel larger than this, no interpolation
    // Raycast
    constexpr float d_raycast_stride_coarse = d_tsdf_voxel_size * 3.f;
    constexpr float d_raycast_stride_fine = d_tsdf_voxel_size * 0.1f;
    constexpr unsigned d_max_strid_step = d_far_clip / d_raycast_stride_coarse;
}