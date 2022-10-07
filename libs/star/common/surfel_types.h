#pragma once
#include <star/common/common_types.h>
#include <star/common/types/typeX.h>

namespace star
{
	/* Helper struct to record the pixel
	 */
	struct PixelCoordinate
	{
		unsigned row;
		unsigned col;
		__host__ __device__ PixelCoordinate() : row(0), col(0) {}
		__host__ __device__ PixelCoordinate(const unsigned row_, const unsigned col_)
			: row(row_), col(col_) {}

		__host__ __device__ const unsigned &x() const { return col; }
		__host__ __device__ const unsigned &y() const { return row; }
		__host__ __device__ unsigned &x() { return col; }
		__host__ __device__ unsigned &y() { return row; }
	};

	/**
	 * \brief The struct for surfel built from depth image
	 *        Shoud be accessed on device.
	 * \member pixel_coord where the surfel is from
	 * \member vertex_confid (x, y, z) is the position in camera frame,
	 *         (w) is the confidence value.
	 * \member normal_radius (x, y, z) is the normalized normal orientation.
	 *         (w) is the radius
	 * \member color_time (x) is float encoded rgb value; (z) is last observed time;
	 *         (w) is the init time
	 */
	struct DepthSurfel
	{
		PixelCoordinate pixel_coord;
		float4 vertex_confid;
		float4 normal_radius;
		float4 color_time;
	};

	// Patch version of knn&weight
	template <unsigned surfel_patch_size>
	struct KNNAndWeight
	{
		ushortX<surfel_patch_size> knn;
		floatX<surfel_patch_size> spatial_weight;
		floatX<surfel_patch_size> connect_weight;

		__host__ __device__ void set_invalid()
		{
#pragma unroll
			for (auto i = 0; i < surfel_patch_size; ++i)
			{
				knn[i] = 0xffff;
				spatial_weight[i] = 0.f;
				connect_weight[i] = 0.f;
			}
		}
	};
}