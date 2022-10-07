#pragma once
#include <star/common/common_types.h>
#include <star/common/ArrayView.h>
#include <star/common/SyncArray.h>
#include <star/common/algorithm_types.h>
#include <star/geometry/sampling/VoxelSubsampler.h>
#include <memory>

namespace star
{
	class VoxelSubsamplerSorting : public VoxelSubsampler
	{
	public:
		using Ptr = std::shared_ptr<VoxelSubsamplerSorting>;
		STAR_DEFAULT_CONSTRUCT_DESTRUCT(VoxelSubsamplerSorting);

		// Again, explicit malloc
		void AllocateBuffer(unsigned max_input_points) override;
		void ReleaseBuffer() override;

		// The main interface
		GArrayView<float4> PerformSubsample(
			const GArrayView<float4> &points,
			const float voxel_size,
			cudaStream_t stream = 0) override;

		// Assume PRE-ALLOCATRED buffer and the
		// AllocateBuffer has been invoked
		void PerformSubsample(
			const GArrayView<float4> &points,
			SyncArray<float4> &subsampled_points,
			const float voxel_size,
			cudaStream_t stream = 0) override;

		/* Take the input as points, build voxel key for each point
		 * Assume the m_point_key is in correct size
		 */
	private:
		GBufferArray<int> m_point_key;
		void buildVoxelKeyForPoints(const GArrayView<float4> &points, const float voxel_size, cudaStream_t stream = 0);

		/* Perform sorting and compaction on the voxel key
		 */
		KeyValueSort<int, float4> m_point_key_sort;
		GBufferArray<unsigned> m_voxel_label;
		PrefixSum m_voxel_label_prefixsum;
		GBufferArray<int> m_compacted_voxel_key;
		GBufferArray<int> m_compacted_voxel_offset;
		void sortCompactVoxelKeys(const GArrayView<float4> &points, cudaStream_t stream = 0);

		/* Collect the subsampled point given the compacted offset
		 */
		GBufferArray<float4> m_subsampled_point; // Optional, for output if no buffer is provided
		void collectSubsampledPoint(
			GBufferArray<float4> &subsampled_points,
			const float voxel_size,
			cudaStream_t stream = 0);
		// Collected the subsampled points and sync it to host
		void collectSynchronizeSubsampledPoint(
			SyncArray<float4> &subsampled_points,
			const float voxel_size,
			cudaStream_t stream = 0);
	};
}