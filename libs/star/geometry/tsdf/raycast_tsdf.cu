#include <star/geometry/tsdf/raycast_tsdf.h>
#include <star/geometry/tsdf/tsdf_constants.h>
#include <star/math/device_mat.h>
#include <cuda_fp16.h>
#include <device_launch_parameters.h>

namespace star { namespace device {
    
    __global__ void RayCastTsdfKernel(
        cudaTextureObject_t tsdf_val,
        cudaTextureObject_t tsdf_weight,
        cudaSurfaceObject_t vertex_confid_raycast,
        const Intrinsic intrinsic,
        const mat34 cam2world,
        const unsigned img_rows,
        const unsigned img_cols,
        const unsigned width,
        const unsigned height,
        const unsigned depth,
        const float voxel_size,
        const float3 origin
    ) {
        const int x = threadIdx.x + blockDim.x * blockIdx.x;
        const int y = threadIdx.y + blockDim.y * blockIdx.y;
        if (x >= img_cols || y >= img_rows) return;

        float3 camera_dir = make_float3(
            (float(x) - intrinsic.principal_x) / intrinsic.focal_x,
            (float(y) - intrinsic.principal_y) / intrinsic.focal_y,
            1.f
        );
        
        float tsdf_front_last = 1.f;
        float3 ray_front;
        float depth_now = d_near_clip;
        float depth_last = depth_now;

        // Coarse-search
        bool coarse_found = false;
        for (auto step = 0; step < d_max_strid_step; ++step) {
            // Step forward
            depth_now += d_raycast_stride_coarse;
            ray_front = cam2world.trans + cam2world.rot * camera_dir * depth_now;
            // Parse location
            float v_x = (ray_front.x - origin.x) / voxel_size;
            float v_y = (ray_front.y - origin.y) / voxel_size;
            float v_z = (ray_front.z - origin.z) / voxel_size;
            if (v_x >= 0.f && v_x < float(width - 1) && 
                v_y >= 0.f && v_y < float(height - 1) && 
                v_z >= 0.f && v_z < float(depth - 1)) {
                float tsdf_front = tex3D<float>(tsdf_val, v_x + 0.5f, v_y + 0.5f, v_z + 0.5f);
                unsigned char weight_front = tex3D<unsigned char>(tsdf_weight, v_x + 0.5f, v_y + 0.5f, v_z + 0.5f);
                if (tsdf_front < 0.f && tsdf_front_last > 0.f) {  // tsdf == 0 is not considered (Regarded as not updated)
                    coarse_found = true;
                    break;
                }
                tsdf_front_last = tsdf_front;
            }
            // Update last info
            depth_last = depth_now;
        }

        // Fine search
        if (coarse_found) {
            float depth_est = 0.f;
            unsigned fine_strid_step = __float2uint_rd(d_raycast_stride_coarse / d_raycast_stride_fine + 1.f);
            depth_now = depth_last;  // Go back to the search start
            for (auto step = 0; step < fine_strid_step; ++step) {
                // Step forward
                depth_now += d_raycast_stride_fine;
                ray_front = cam2world.trans + cam2world.rot * camera_dir * depth_now;
                // Parse location
                float v_x = (ray_front.x - origin.x) / voxel_size;
                float v_y = (ray_front.y - origin.y) / voxel_size;
                float v_z = (ray_front.z - origin.z) / voxel_size;
                if (v_x >= 0.f && v_x < float(width - 1) &&
                    v_y >= 0.f && v_y < float(height - 1) &&
                    v_z >= 0.f && v_z < float(depth - 1)) {
                    float tsdf_front = tex3D<float>(tsdf_val, v_x + 0.5f, v_y + 0.5f, v_z + 0.5f);
                    unsigned char weight_front = tex3D<unsigned char>(tsdf_weight, v_x + 0.5f, v_y + 0.5f, v_z + 0.5f);
                    if (tsdf_front < 0.f && tsdf_front_last > 0.f) {  // tsdf == 0 is not considered (Regarded as not updated)
                        depth_est = (tsdf_front * depth_now - tsdf_front_last * depth_last) / (tsdf_front - tsdf_front_last);
                        break;
                    }
                    tsdf_front_last = tsdf_front;
                }
                // Update last info
                depth_last = depth_now;
            }
            float4 vertex_confid;
            vertex_confid.x = (float(x) - intrinsic.principal_x) / intrinsic.focal_x * depth_est;
            vertex_confid.y = (float(y) - intrinsic.principal_y) / intrinsic.focal_y * depth_est;
            vertex_confid.z = depth_est;
            vertex_confid.w = 1.f;  // Currently using a fixed confidence
            surf2Dwrite(vertex_confid, vertex_confid_raycast, x * sizeof(float4), y);
        }
        else {
            surf2Dwrite(make_float4(0.f, 0.f, 0.f, 1.f), vertex_confid_raycast, x * sizeof(float4), y);
        }
    }
}
}


void star::RayCastTsdf(
    const Tsdf& tsdf,
    cudaSurfaceObject_t vertex_confid_raycast,
    const Intrinsic& intrinsic,
    const Eigen::Matrix4f& cam2world,
    const unsigned img_rows,
    const unsigned img_cols,
    cudaStream_t stream
) {
    dim3 blk(16, 16);
    dim3 grid(divUp(img_cols, blk.x), divUp(img_rows, blk.y));
    device::RayCastTsdfKernel<<<grid, blk, 0, stream>>>(
        tsdf.tsdf_val.texture,
        tsdf.tsdf_weight.texture,
        vertex_confid_raycast,
        intrinsic,
        mat34(cam2world),
        img_rows,
        img_cols,
        tsdf.width,
        tsdf.height,
        tsdf.depth,
        tsdf.voxel_size,
        tsdf.origin
    );

    //Sync and check error
#if defined(CUDA_DEBUG_SYNC_CHECK)
    cudaSafeCall(cudaStreamSynchronize(stream));
    cudaSafeCall(cudaGetLastError());
#endif
}