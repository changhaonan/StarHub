#pragma once
#include <star/common/common_types.h>
#include <star/common/surfel_types.h>

namespace star
{
	void markValidDepthPixel(
		cudaTextureObject_t vertex_img,
		const unsigned rows, const unsigned cols,
		GArray<char> &valid_indicator,
		cudaStream_t stream = 0);

	void collectDepthSurfel(
		cudaTextureObject_t vertex_confid_map,
		cudaTextureObject_t normal_radius_map,
		cudaTextureObject_t color_time_map,
		const GArray<int> &selected_array,
		const unsigned rows, const unsigned cols,
		GArray<DepthSurfel> &valid_depth_surfel,
		cudaStream_t stream = 0);
}