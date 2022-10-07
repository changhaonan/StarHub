#include <star/math/vector_ops.hpp>
#include <star/geometry/tsdf/marching_cube.h>
#include <device_launch_parameters.h>

namespace star::device
{
    __device__ __forceinline__ void vertexInterpolate(float4 &p, const float4 &p0, const float4 &p1, const float f0, const float f1)
    {
        float t = (0.f - f0) / (f1 - f0 + 1e-6f);
        t = ((f0 == 0.f) && (f1 == 0.f)) ? 0.5f : t;
        p.x = p0.x + t * (p1.x - p0.x);
        p.y = p0.y + t * (p1.y - p0.y);
        p.z = p0.z + t * (p1.z - p0.z);
        p.w = 1.f;
    }

    __device__ __forceinline__ float readTsdf(
        cudaTextureObject_t tsdf_val,
        cudaTextureObject_t tsdf_weight,
        const int3 dim_3d, const int x, const int y, const int z, float &weight)
    {
        if (x >= (dim_3d.x - 1) || y >= (dim_3d.y - 1) || z >= (dim_3d.z - 1) || x < 0 || y < 0 || z < 0)
        { // Leave enough gap
            weight = 0.f;
            return 0.f;
        }
        else
        {
            unsigned char weight_unchar = tex3D<unsigned char>(tsdf_weight, float(x) + 0.5f, float(y) + 0.5f, float(z) + 0.5f);
            weight = float(weight_unchar);

            float tsdf = tex3D<float>(tsdf_val, float(x) + 0.5f, float(y) + 0.5f, float(z) + 0.5f); // 0.5f is the texture offset
            return tsdf;
        }
    }

    // compute cube index
    __device__ __forceinline__ int computeCubeIndex(
        cudaTextureObject_t tsdf_val,
        cudaTextureObject_t tsdf_weight,
        const int3 dim_3d, const int x, const int y, const int z, float tsdf_values[8])
    {

        float weight;
        int cube_index = 0; // calculate flag indicating if each vertex is inside or outside isosurface

        // remember to check out-range
        cube_index += static_cast<int>(tsdf_values[0] = readTsdf(tsdf_val, tsdf_weight, dim_3d, x, y, z, weight) < 0.f);
        if (weight == 0.f)
            return 0;
        cube_index += static_cast<int>(tsdf_values[1] = readTsdf(tsdf_val, tsdf_weight, dim_3d, x + 1, y, z, weight) < 0.f) << 1;
        if (weight == 0.f)
            return 0;
        cube_index += static_cast<int>(tsdf_values[2] = readTsdf(tsdf_val, tsdf_weight, dim_3d, x + 1, y + 1, z, weight) < 0.f) << 2;
        if (weight == 0.f)
            return 0;
        cube_index += static_cast<int>(tsdf_values[3] = readTsdf(tsdf_val, tsdf_weight, dim_3d, x, y + 1, z, weight) < 0.f) << 3;
        if (weight == 0.f)
            return 0;
        cube_index += static_cast<int>(tsdf_values[4] = readTsdf(tsdf_val, tsdf_weight, dim_3d, x, y, z + 1, weight) < 0.f) << 4;
        if (weight == 0.f)
            return 0;
        cube_index += static_cast<int>(tsdf_values[5] = readTsdf(tsdf_val, tsdf_weight, dim_3d, x + 1, y, z + 1, weight) < 0.f) << 5;
        if (weight == 0.f)
            return 0;
        cube_index += static_cast<int>(tsdf_values[6] = readTsdf(tsdf_val, tsdf_weight, dim_3d, x + 1, y + 1, z + 1, weight) < 0.f) << 6;
        if (weight == 0.f)
            return 0;
        cube_index += static_cast<int>(tsdf_values[7] = readTsdf(tsdf_val, tsdf_weight, dim_3d, x, y + 1, z + 1, weight) < 0.f) << 7;
        if (weight == 0.f)
            return 0;

        return cube_index;
    }

    __device__ __forceinline__ void readCubeTsdf(
        cudaTextureObject_t tsdf_val,
        cudaTextureObject_t tsdf_weight,
        const int3 dim_3d,
        const const int x, const int y, const int z,
        float tsdf[8])
    {
        float weight;
        tsdf[0] = readTsdf(tsdf_val, tsdf_weight, dim_3d, x, y, z, weight);
        tsdf[1] = readTsdf(tsdf_val, tsdf_weight, dim_3d, x + 1, y, z, weight);
        tsdf[2] = readTsdf(tsdf_val, tsdf_weight, dim_3d, x + 1, y + 1, z, weight);
        tsdf[3] = readTsdf(tsdf_val, tsdf_weight, dim_3d, x, y + 1, z, weight);
        tsdf[4] = readTsdf(tsdf_val, tsdf_weight, dim_3d, x, y, z + 1, weight);
        tsdf[5] = readTsdf(tsdf_val, tsdf_weight, dim_3d, x + 1, y, z + 1, weight);
        tsdf[6] = readTsdf(tsdf_val, tsdf_weight, dim_3d, x + 1, y + 1, z + 1, weight);
        tsdf[7] = readTsdf(tsdf_val, tsdf_weight, dim_3d, x, y + 1, z + 1, weight);
    }

    __global__ void PreInterploateAtomicKernel(
        cudaTextureObject_t tsdf_val,
        cudaTextureObject_t tsdf_weight,
        GArraySlice<float4> pre_itp_vertex,
        GArraySlice<int> index_edge,
        const int3 dim_3d,
        const float3 origin,
        const float voxel_size,
        int *count)
    {
        const int x = threadIdx.x + blockDim.x * blockIdx.x;
        const int y = threadIdx.y + blockDim.y * blockIdx.y;
        const int z = threadIdx.z + blockDim.z * blockIdx.z;
        if (x >= dim_3d.x || y >= dim_3d.y || z >= dim_3d.z)
            return;
        const auto v_id = x * dim_3d.y * dim_3d.z + y * dim_3d.z + z;

        float tsdf_values[4];
        float weight;

        float tsdf = readTsdf(tsdf_val, tsdf_weight, dim_3d, x, y, z, weight);
        float tsdf_other;
        tsdf_values[1] =
            tsdf_values[2] = readTsdf(tsdf_val, tsdf_weight, dim_3d, x, y + 1, z, weight);
        tsdf_values[3] = readTsdf(tsdf_val, tsdf_weight, dim_3d, x, y, z + 1, weight);

        float4 tsdf_center = make_float4(
            origin.x + voxel_size * float(x),
            origin.x + voxel_size * float(y),
            origin.x + voxel_size * float(z),
            1.f);

        // Interploation between x & x + 1
        tsdf_other = readTsdf(tsdf_val, tsdf_weight, dim_3d, x + 1, y, z, weight);
        if (tsdf * tsdf_other < 0)
        {
            float4 tsdf_center_other = make_float4(
                tsdf_center.x + voxel_size,
                tsdf_center.y,
                tsdf_center.z,
                1.f);
            int old_count = atomicAdd(count, 1);
            vertexInterpolate(
                pre_itp_vertex[old_count],
                tsdf_center, tsdf_center_other,
                tsdf, tsdf_other);
            index_edge[3 * v_id + 0] = old_count;
        }
        else
        {
            index_edge[3 * v_id + 0] = -1;
        }
        // Interpolation between y & y + 1
        tsdf_other = readTsdf(tsdf_val, tsdf_weight, dim_3d, x, y + 1, z, weight);
        if (tsdf * tsdf_other < 0)
        {
            float4 tsdf_center_other = make_float4(
                tsdf_center.x,
                tsdf_center.y + voxel_size,
                tsdf_center.z,
                1.f);
            int old_count = atomicAdd(count, 1);
            vertexInterpolate(
                pre_itp_vertex[old_count],
                tsdf_center, tsdf_center_other,
                tsdf, tsdf_other);
            index_edge[3 * v_id + 1] = old_count;
        }
        else
        {
            index_edge[3 * v_id + 1] = -1;
        }
        // Interpolation between z & z + 1
        tsdf_other = readTsdf(tsdf_val, tsdf_weight, dim_3d, x, y, z + 1, weight);
        if (tsdf * tsdf_other < 0)
        {
            float4 tsdf_center_other = make_float4(
                tsdf_center.x,
                tsdf_center.y,
                tsdf_center.z + voxel_size,
                1.f);
            int old_count = atomicAdd(count, 1);
            vertexInterpolate(
                pre_itp_vertex[old_count],
                tsdf_center, tsdf_center_other,
                tsdf, tsdf_other);
            index_edge[3 * v_id + 2] = old_count;
        }
        else
        {
            index_edge[3 * v_id + 2] = -1;
        }
    }

    __global__ void GetOccupiedVoxels(
        cudaTextureObject_t tsdf_val,
        cudaTextureObject_t tsdf_weight,
        unsigned *num_vertices,
        GArraySlice<int> index_voxel,
        GArraySlice<int> index_cube,
        const int *number_vertices_table,
        const int3 dim_3d,
        unsigned *count)
    {
        const int x = threadIdx.x + blockDim.x * blockIdx.x;
        const int y = threadIdx.y + blockDim.y * blockIdx.y;
        const int z = threadIdx.z + blockDim.z * blockIdx.z;
        if (x >= dim_3d.x || y >= dim_3d.y || z >= dim_3d.z)
            return;
        const auto v_id = x * dim_3d.y * dim_3d.z + y * dim_3d.z + z; // encoding

        float tsdf_values[8];
        int cube_index = computeCubeIndex(tsdf_val, tsdf_weight, dim_3d, x, y, z, tsdf_values);

        int num_vertices_now = (cube_index == 0 || cube_index == 255) ? 0 : number_vertices_table[cube_index];

        // compute x, y, z
        if (num_vertices_now > 0)
        {
            int count_old = atomicAdd(count, unsigned(1));
            num_vertices[count_old] = unsigned(num_vertices_now);
            // update index
            index_voxel[count_old] = v_id;
            index_cube[count_old] = cube_index;
        }
    }

    __global__ void ComputeTriangleKernel(
        cudaTextureObject_t tsdf_val,
        cudaTextureObject_t tsdf_weight,
        GArrayView<int> index_voxel,
        GArrayView<int> index_cube,
        GArraySlice<int> triangle_faces,
        GArraySlice<float4> triangle_vertices,
        const unsigned *num_vertices_offset,
        const unsigned num_occupied_voxel,
        const int *triangle_table,
        const int3 dim_3d,
        const float voxel_size,
        const float3 origin)
    {
        const auto idx = threadIdx.x + blockDim.x * blockIdx.x;
        if (idx >= num_occupied_voxel)
            return;
        const unsigned offset = (idx == 0) ? 0 : num_vertices_offset[idx - 1];
        const unsigned num_vertices = (num_vertices_offset[idx] - offset);

        int cube_index = index_cube[idx];
        int voxel_index = index_voxel[idx]; // Parse v_x, v_y, v_z
        int v_z = (voxel_index) % dim_3d.z;
        int v_xy = (voxel_index) / dim_3d.z;
        int v_y = v_xy % dim_3d.y;
        int v_x = v_xy / dim_3d.y;

        float tsdf[8];
        readCubeTsdf(tsdf_val, tsdf_weight, dim_3d, v_x, v_y, v_z, tsdf);

        if (num_vertices == 0)
        {
            printf(
                "idx: %d, num_occupied: %d.\n", idx, num_occupied_voxel);
        }

        // Interpolation
        for (auto i = 0; i < num_vertices; ++i)
        { // Go over
            int itp_vertex_id = triangle_table[16 * cube_index + i];
            float4 vertex_0 = make_float4(
                origin.x + float(v_x) * voxel_size,
                origin.y + float(v_y) * voxel_size,
                origin.z + float(v_z) * voxel_size,
                1.f);
            float4 vertex_1 = make_float4(
                origin.x + float(v_x) * voxel_size,
                origin.y + float(v_y) * voxel_size,
                origin.z + float(v_z) * voxel_size,
                1.f);
            float tsdf_0, tsdf_1;
            if (itp_vertex_id == 0)
            {
                vertex_0;
                vertex_1.x += voxel_size;
                tsdf_0 = tsdf[0];
                tsdf_1 = tsdf[1];
            }
            else if (itp_vertex_id == 1)
            {
                vertex_0.x += voxel_size;
                vertex_1.x += voxel_size;
                vertex_1.y += voxel_size;
                tsdf_0 = tsdf[1];
                tsdf_1 = tsdf[2];
            }
            else if (itp_vertex_id == 2)
            {
                vertex_0.x += voxel_size;
                vertex_0.y += voxel_size;
                vertex_1.y += voxel_size;
                tsdf_0 = tsdf[2];
                tsdf_1 = tsdf[3];
            }
            else if (itp_vertex_id == 3)
            {
                vertex_0.y += voxel_size;
                vertex_1;
                tsdf_0 = tsdf[3];
                tsdf_1 = tsdf[0];
            }
            else if (itp_vertex_id == 4)
            {
                vertex_0.z += voxel_size;
                vertex_1.x += voxel_size;
                vertex_1.z += voxel_size;
                tsdf_0 = tsdf[4];
                tsdf_1 = tsdf[5];
            }
            else if (itp_vertex_id == 5)
            {
                vertex_0.x += voxel_size;
                vertex_0.z += voxel_size;
                vertex_1.x += voxel_size;
                vertex_1.y += voxel_size;
                vertex_1.z += voxel_size;
                tsdf_0 = tsdf[5];
                tsdf_1 = tsdf[6];
            }
            else if (itp_vertex_id == 6)
            {
                vertex_0.x += voxel_size;
                vertex_0.y += voxel_size;
                vertex_0.z += voxel_size;
                vertex_1.y += voxel_size;
                vertex_1.z += voxel_size;
                tsdf_0 = tsdf[6];
                tsdf_1 = tsdf[7];
            }
            else if (itp_vertex_id == 7)
            {
                vertex_0.y += voxel_size;
                vertex_0.z += voxel_size;
                vertex_1.z += voxel_size;
                tsdf_0 = tsdf[7];
                tsdf_1 = tsdf[4];
            }
            else if (itp_vertex_id == 8)
            {
                vertex_0;
                vertex_1.z += voxel_size;
                tsdf_0 = tsdf[0];
                tsdf_1 = tsdf[4];
            }
            else if (itp_vertex_id == 9)
            {
                vertex_0.x += voxel_size;
                vertex_1.x += voxel_size;
                vertex_1.z += voxel_size;
                tsdf_0 = tsdf[1];
                tsdf_1 = tsdf[5];
            }
            else if (itp_vertex_id == 10)
            {
                vertex_0.x += voxel_size;
                vertex_0.y += voxel_size;
                vertex_1.x += voxel_size;
                vertex_1.y += voxel_size;
                vertex_1.z += voxel_size;
                tsdf_0 = tsdf[2];
                tsdf_1 = tsdf[6];
            }
            else if (itp_vertex_id == 11)
            {
                vertex_0.y += voxel_size;
                vertex_1.y += voxel_size;
                vertex_1.z += voxel_size;
                tsdf_0 = tsdf[3];
                tsdf_1 = tsdf[7];
            }

            float4 itp_vertex;
            vertexInterpolate(itp_vertex, vertex_0, vertex_1, tsdf_0, tsdf_1);
            triangle_vertices[offset + i] = itp_vertex;
            triangle_faces[offset + i] = int(offset) + i;
        }
    }
}

void star::ComputeOccupiedVoxels(
    const Tsdf &tsdf,
    GArraySlice<int> index_voxel,
    GArraySlice<int> index_cube, // Cube type encoding, Size: num of occupied voxel
    unsigned *num_vertices,      // Number of vertex in each voxel, used for prefixsum
    unsigned &num_occupied_voxel,
    const int *number_vertices_table,
    cudaStream_t stream)
{

    int3 dim_3d = make_int3(tsdf.width, tsdf.height, tsdf.depth);
    unsigned *num_occupied;
    cudaSafeCall(cudaMallocAsync((void **)&num_occupied, sizeof(unsigned), stream));
    cudaSafeCall(cudaMemsetAsync(num_occupied, 0, sizeof(unsigned), stream));

    dim3 blk(8, 8, 8);
    dim3 grid(divUp(dim_3d.x, blk.x), divUp(dim_3d.y, blk.y), divUp(dim_3d.z, blk.z));
    device::GetOccupiedVoxels<<<grid, blk, 0, stream>>>(
        tsdf.tsdf_val.texture,
        tsdf.tsdf_weight.texture,
        num_vertices,
        index_voxel,
        index_cube,
        number_vertices_table,
        dim_3d,
        num_occupied);

    // Sync and check error
#if defined(CUDA_DEBUG_SYNC_CHECK)
    cudaSafeCall(cudaStreamSynchronize(stream));
    cudaSafeCall(cudaGetLastError());
#endif

    cudaSafeCall(cudaMemcpyAsync(&num_occupied_voxel, num_occupied, sizeof(unsigned), cudaMemcpyDeviceToHost, stream));
    cudaSafeCall(cudaFreeAsync(num_occupied, stream));
    printf("Marching cubes has %d voxel.\n", num_occupied_voxel);
}

void star::ComputeTriangle(
    const Tsdf &tsdf,
    GArrayView<int> index_voxel,
    GArrayView<int> index_cube,
    GArraySlice<float4> triangle_vertices,
    GArraySlice<int> triangle_faces,
    const unsigned *num_vertices,
    const unsigned *num_vertices_offset,
    const unsigned num_occupied_voxel,
    const int *triangle_table,
    cudaStream_t stream)
{
    int3 dim_3d = make_int3(tsdf.width, tsdf.height, tsdf.depth);
    dim3 blk(128);
    dim3 grid(divUp(num_occupied_voxel, blk.x));
    device::ComputeTriangleKernel<<<grid, blk, 0, stream>>>(
        tsdf.tsdf_val.texture,
        tsdf.tsdf_weight.texture,
        index_voxel,
        index_cube,
        triangle_faces,
        triangle_vertices,
        num_vertices_offset,
        num_occupied_voxel,
        triangle_table,
        dim_3d,
        tsdf.voxel_size,
        tsdf.origin);

    // Sync and check error
#if defined(CUDA_DEBUG_SYNC_CHECK)
    cudaSafeCall(cudaStreamSynchronize(stream));
    cudaSafeCall(cudaGetLastError());
#endif
}

void star::MarchingCube(
    const Tsdf &tsdf,
    GBufferArray<float4> &extracted_surfel_confid,
    GBufferArray<float4> &extracted_normal_radius,
    GBufferArray<float4> &extracted_color_time,
    GBufferArray<int> &extracted_faces, // Triangle surfaces in vertex id
    GBufferArray<int> &index_voxel_buffer,
    GBufferArray<int> &index_cube_buffer,
    PrefixSum &num_vertices_prefixsum,
    GArray<unsigned> &num_vertices,
    const int *triangle_table,
    const int *number_vertices_table,
    cudaStream_t stream)
{
    unsigned num_occupied_voxel;
    unsigned num_triangle_vertices;
    // Compute Occupued voxels first & num of vertices
    ComputeOccupiedVoxels(
        tsdf,
        index_voxel_buffer.Slice(),
        index_cube_buffer.Slice(), // Cube type encoding, Size: num of occupied voxel
        num_vertices,              // Number of vertex in each voxel, used for prefixsum
        num_occupied_voxel,
        number_vertices_table,
        stream);

    cudaSafeCall(cudaStreamSynchronize(stream)); // Synchronize device & host before use
    // Resize
    index_voxel_buffer.ResizeArrayOrException(num_occupied_voxel);
    index_cube_buffer.ResizeArrayOrException(num_occupied_voxel);
    // Prefixsum computing offset
    num_vertices_prefixsum.InclusiveSum(num_vertices, stream);
    cudaSafeCall(cudaMemcpyAsync(&num_triangle_vertices, num_vertices_prefixsum.valid_prefixsum_array.ptr() + num_occupied_voxel - 1,
                                 sizeof(unsigned), cudaMemcpyDeviceToHost));
    cudaSafeCall(cudaStreamSynchronize(stream)); // Synchronize device & host before use
    extracted_surfel_confid.ResizeArrayOrException(num_triangle_vertices);
    extracted_normal_radius.ResizeArrayOrException(num_triangle_vertices);
    extracted_color_time.ResizeArrayOrException(num_triangle_vertices);
    extracted_faces.ResizeArrayOrException(num_triangle_vertices);

    ComputeTriangle(
        tsdf,
        index_voxel_buffer.View(),
        index_cube_buffer.View(),
        extracted_surfel_confid.Slice(),
        extracted_faces.Slice(),
        num_vertices,
        num_vertices_prefixsum.valid_prefixsum_array,
        num_occupied_voxel,
        triangle_table,
        stream);
    cudaSafeCall(cudaStreamSynchronize(stream)); // Synchronize device & host before use, we are doing debug, time is fine
    printf("Marching cubes has %d triangles.\n", num_triangle_vertices);
}