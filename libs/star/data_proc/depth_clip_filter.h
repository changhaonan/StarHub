#pragma once
#include <star/common/common_types.h>
#include <star/common/common_utils.h>
#include <star/math/device_mat.h>

namespace star
{
	/**
	 * \brief Take a raw image as input, do bilateral filtering
	 * \param raw_depth (raw_img_rows, raw_img_cols)
	 * \param filter_depth (clip_img_rows, clip_img_cols)
	 * \param clip_img_rows, clip_img_cols in pixels
	 * \param clip_near, clip_far in [mm],
	 * \param enable gaussian filter or not (Only depth filter will be enabled)
	 */
	void clipFilterDepthImage(
		cudaTextureObject_t raw_depth,
		const unsigned clip_img_rows, const unsigned clip_img_cols,
		const unsigned clip_near, const unsigned clip_far,
		cudaSurfaceObject_t filter_depth,
		const bool enable_gaussian = true,
		const float sigma_s = 4.5f,
		const float sigma_r = 30.f,
		cudaStream_t stream = 0);

	void reprojectDepthToRGB(
		cudaTextureObject_t raw_depth,
		cudaSurfaceObject_t reprojected_depth,
		GArray2D<unsigned short> reprojected_buffer,
		const unsigned raw_rows, const unsigned raw_cols,
		const IntrinsicInverse &raw_depth_intrinsic_inverse,
		const Intrinsic &raw_rgb_intrinsic,
		const mat34 &depth2rgb,
		const int factor = 2,
		cudaStream_t stream = 0);
}