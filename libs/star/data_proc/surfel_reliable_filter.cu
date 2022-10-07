#include <device_launch_parameters.h>
#include <star/math/vector_ops.hpp>
#include <star/math/eigen33.h>
#include <star/data_proc/surfel_reliable_filter.h>

namespace star
{
    namespace device
    {

        // constexpr float d_reliable_angle_thresh = 0.2f;  // Strict constraints
        constexpr float d_reliable_angle_thresh = 0.0f; // Loose constraints
        constexpr float d_back_ground_depth = 10.f;     // Drag invalid pixel out of range during interpolation
        constexpr float d_invalid_depth = 0.f;

        __device__ __forceinline__
            float4
            get_vertex_confid(
                cudaTextureObject_t raw_depth_map, const float x, const float y, const IntrinsicInverse &intrinsic_inv)
        {
            const unsigned short raw_depth = tex2D<unsigned short>(raw_depth_map, x + 0.5f, y + 0.5f);
            float4 vertex_confid;

            // scale the depth to [m]
            // The depth image is always in [mm]
            vertex_confid.z = float(raw_depth) / (1000.f);
            vertex_confid.x = (x - intrinsic_inv.principal_x) * intrinsic_inv.inv_focal_x * vertex_confid.z;
            vertex_confid.y = (y - intrinsic_inv.principal_y) * intrinsic_inv.inv_focal_y * vertex_confid.z;
            vertex_confid.w = 1.f; // Or other
            return vertex_confid;
        }

        enum
        {
            window_dim = 7,
            halfsize = 3,
            window_size = window_dim * window_dim
        };

        __global__ void filterUnreliableDepthKernel(
            cudaTextureObject_t raw_depth_map,
            cudaSurfaceObject_t filtered_depth_map,
            const unsigned rows, const unsigned cols,
            const float clip_near, const float clip_far,
            const IntrinsicInverse intrinsic_inv)
        {
            // The index on the raw map
            const auto x = threadIdx.x + blockIdx.x * blockDim.x;
            const auto y = threadIdx.y + blockIdx.y * blockDim.y;
            if (x >= cols || y >= rows)
                return;

            bool reliable = true;
            bool is_background = false;
            unsigned raw_depth = tex2D<unsigned>(raw_depth_map, x, y);
            // Depth filter
            if (abs(float(raw_depth) / 1000.f) < clip_near)
            {
                reliable = false; // Is too near
            }
            if (abs(float(raw_depth) / 1000.f) > clip_far)
            {
                is_background = true; // Is too far
            }
            else
            {
                // Direction filter
                // Estimate the normal
                float3 normal_center = make_float3(0, 0, 0);

                // The vertex at the center
                const float4 vertex_center = get_vertex_confid(raw_depth_map, x, y, intrinsic_inv);
                if (!is_zero_vertex(vertex_center))
                {
                    float4 centeroid = make_float4(0, 0, 0, 0);
                    int counter = 0;
                    // First window search to determine the center
                    for (int cy = y - halfsize; cy <= y + halfsize; cy += 1)
                    {
                        for (int cx = x - halfsize; cx <= x + halfsize; cx += 1)
                        {
                            const float4 p = get_vertex_confid(raw_depth_map, cx, cy, intrinsic_inv);
                            if (!is_zero_vertex(p))
                            {
                                centeroid.x += p.x;
                                centeroid.y += p.y;
                                centeroid.z += p.z;
                                counter++;
                            }
                        }
                    } // End of first window search

                    // At least half of the window is valid
                    if (counter > (window_size / 2))
                    {
                        centeroid *= (1.0f / counter);
                        float covariance[6] = {0};

                        // Second window search to compute the normal
                        for (int cy = y - halfsize; cy < y + halfsize; cy += 1)
                        {
                            for (int cx = x - halfsize; cx < x + halfsize; cx += 1)
                            {
                                const float4 p = get_vertex_confid(raw_depth_map, cx, cy, intrinsic_inv);
                                if (!is_zero_vertex(p))
                                {
                                    const float4 diff = p - centeroid;
                                    // Compute the covariance
                                    covariance[0] += diff.x * diff.x; //(0, 0)
                                    covariance[1] += diff.x * diff.y; //(0, 1)
                                    covariance[2] += diff.x * diff.z; //(0, 2)
                                    covariance[3] += diff.y * diff.y; //(1, 1)
                                    covariance[4] += diff.y * diff.z; //(1, 2)
                                    covariance[5] += diff.z * diff.z; //(2, 2)
                                }
                            }
                        } // End of second window search

                        // The eigen value for normal
                        eigen33 eigen(covariance);
                        float3 normal;
                        eigen.compute(normal);
                        if (dotxyz(normal, vertex_center) >= 0.0f)
                            normal *= -1;

                        normal_center = normal;
                    } // End of check the number of valid pixels
                }     // If the vertex is non-zero

                float3 camera_direction = make_float3(vertex_center.x, vertex_center.y, vertex_center.z);
                float camera_direction_norm = norm(camera_direction);
                camera_direction = make_float3(
                    camera_direction.x / (1e-6f + camera_direction_norm),
                    camera_direction.y / (1e-6f + camera_direction_norm),
                    camera_direction.z / (1e-6f + camera_direction_norm));
                float3 normal = make_float3(normal_center.x, normal_center.y, normal_center.z);
                float normal_norm = norm(normal);
                normal = make_float3(
                    normal.x / (1e-6f + normal_norm),
                    normal.y / (1e-6f + normal_norm),
                    normal.z / (1e-6f + normal_norm));
                float camera_dot = dot(camera_direction, normal);
                // if (camera_dot > -d_reliable_angle_thresh) {  // Unreliable
                //     reliable = false;
                // }
            }

            if (!reliable)
            {
                surf2Dwrite(d_invalid_depth, filtered_depth_map, x * sizeof(float), y); // Remove & set as invalid
            }
            else if (is_background)
            {
                surf2Dwrite(d_back_ground_depth, filtered_depth_map, x * sizeof(float), y); // Remove & set as background
            }
            else
            {
                const unsigned short raw_depth = tex2D<unsigned short>(raw_depth_map, float(x) + 0.5f, float(y) + 0.5f);
                const float raw_depth_float = float(raw_depth) / 1000.f;
                surf2Dwrite(raw_depth_float, filtered_depth_map, x * sizeof(float), y);
            }
        }

    }
}

void star::filterUnreliableDepth(
    cudaTextureObject_t raw_depth_map,
    cudaSurfaceObject_t filtered_depth_map,
    const unsigned rows, const unsigned cols,
    const float clip_near, const float clip_far,
    const Intrinsic &intrinsic,
    cudaStream_t stream)
{

    IntrinsicInverse intrinsic_inv;
    intrinsic_inv.principal_x = intrinsic.principal_x;
    intrinsic_inv.principal_y = intrinsic.principal_y;
    intrinsic_inv.inv_focal_x = 1.f / intrinsic.focal_x;
    intrinsic_inv.inv_focal_y = 1.f / intrinsic.focal_y;

    dim3 blk(16, 16);
    dim3 grid(divUp(cols, blk.x), divUp(rows, blk.y));
    device::filterUnreliableDepthKernel<<<grid, blk, 0, stream>>>(
        raw_depth_map,
        filtered_depth_map,
        rows, cols,
        clip_near, clip_far,
        intrinsic_inv);

    // Sync and check error
#if defined(CUDA_DEBUG_SYNC_CHECK)
    cudaSafeCall(cudaStreamSynchronize(stream));
    cudaSafeCall(cudaGetLastError());
#endif
}