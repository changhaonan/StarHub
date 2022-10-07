/*
 * Created by Haonan Chang, 10/21/2021
 */
#pragma once

#include <star/common/common_types.h>

namespace star
{
	void density_transfer(
		cudaTextureObject_t color_time_map,
		unsigned rows, unsigned cols,
		cudaSurfaceObject_t density_map,
		cudaStream_t stream);

}