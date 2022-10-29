#pragma once
#include <star/common/common_types.h>
#include <star/common/ArrayView.h>

namespace star
{
	void createVertexConfidMap(
		cudaTextureObject_t depth_img,
		const unsigned rows, const unsigned cols,
		const IntrinsicInverse intrinsic_inv,
		cudaSurfaceObject_t vertex_confid_map,
		cudaStream_t stream = 0);

	void createNormalRadiusMap(
		cudaTextureObject_t vertex_map,
		const unsigned rows, const unsigned cols,
		cudaSurfaceObject_t normal_radius_map,
		const float surfel_radius_scale,
		cudaStream_t stream = 0);

	void createColorTimeMap(
		const GArray<uchar3> raw_rgb_img,
		const unsigned clip_rows, const unsigned clip_cols,
		const float init_time,
		cudaSurfaceObject_t color_time_map,
		cudaStream_t stream = 0);

	void createScaledColorTimeMap( // Color time map creation, but scaled
		const GArrayView<uchar3> raw_rgb_img,
		const unsigned raw_rows, const unsigned raw_cols,
		const float scale,
		const float init_time,
		cudaSurfaceObject_t color_time_map,
		cudaStream_t stream = 0);

	void createScaledRGBDMap( // RGBD image, D is the inverse of depth
		const GArrayView<uchar3> raw_rgb_img,
		cudaTextureObject_t filtered_depth_img,
		const unsigned raw_rows, const unsigned raw_cols,
		const float scale,
		const float clip_near, const float clip_far,
		cudaSurfaceObject_t rgbd_map,
		cudaStream_t stream = 0);

	void createScaledOpticalFlowMap( // Optical flow map creation, but scaled
		const float2 *__restrict__ raw_opticalflow_map,
		const unsigned raw_rows, const unsigned raw_cols,
		const float scale,
		cudaSurfaceObject_t opticalflow_map,
		cudaStream_t stream = 0);

	void createNormalizedRGBMap(
		const GArray<uchar3> raw_rgb_img,
		cudaSurfaceObject_t normalized_rgb_map,
		const unsigned rows, const unsigned cols,
		cudaStream_t stream);

	void createScaledDepthMap(
		cudaTextureObject_t filtered_depth_img,
		const unsigned raw_rows, const unsigned raw_cols,
		const float scale,
		cudaSurfaceObject_t scaled_depth_map,
		cudaStream_t stream);
}