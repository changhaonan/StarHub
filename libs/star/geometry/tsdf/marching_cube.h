#pragma once
#include <star/common/common_types.h>
#include <star/common/algorithm_types.h>
#include <star/geometry/tsdf/Tsdf.h>

namespace star
{
    void MarchingCube(
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
        cudaStream_t stream);

    /**
     * \brief Check all voxels. If this voxel has non-zero number of
     * vertex.
     */
    void ComputeOccupiedVoxels(
        const Tsdf &tsdf,
        GArraySlice<int> index_voxel,
        GArraySlice<int> index_cube, // Cube type encoding, Size: num of occupied voxel
        unsigned *num_vertices,      // Number of vertex in each voxel, used for prefixsum
        unsigned &num_occupied_voxel,
        const int *number_vertices_table,
        cudaStream_t stream = 0);

    /**
     * \brief Compact triangle_vertices & triangle faces computed before
     */
    void ComputeTriangle(
        const Tsdf &tsdf,
        GArrayView<int> index_voxel,
        GArrayView<int> index_cube,
        GArraySlice<float4> triangle_vertices,
        GArraySlice<int> triangle_faces,
        const unsigned *num_vertices,
        const unsigned *num_vertices_offset,
        const unsigned num_occupied_voxel,
        const int *triangle_table,
        cudaStream_t stream = 0);

    // TODO: Add compaction method
}