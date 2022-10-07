#pragma once
#include <star/common/macro_utils.h>
#include <star/common/common_types.h>
#include <star/common/ArrayView.h>
#include <star/common/SyncArray.h>
#include <memory>

namespace star
{
	class VoxelSubsampler
	{
	public:
		using Ptr = std::shared_ptr<VoxelSubsampler>;
		VoxelSubsampler() = default;
		virtual ~VoxelSubsampler() = default;
		STAR_NO_COPY_ASSIGN_MOVE(VoxelSubsampler);

		// Again, explicit malloc
		virtual void AllocateBuffer(unsigned max_input_points) = 0;
		virtual void ReleaseBuffer() = 0;

		// The interface functions
		virtual GArrayView<float4> PerformSubsample(
			const GArrayView<float4> &points,
			const float voxel_size,
			cudaStream_t stream = 0) = 0;

		virtual void PerformSubsample(
			const GArrayView<float4> &points,
			SyncArray<float4> &subsampled_points,
			const float voxel_size,
			cudaStream_t stream = 0) = 0;
	};
}