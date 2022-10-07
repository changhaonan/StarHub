#pragma once
#include <star/common/common_types.h>
#include <star/math/device_mat.h>

namespace star
{

	// Twist representation
	struct twist_rotation
	{

		__host__ __device__ twist_rotation(const float3 &_xi) { xi = _xi; }
		__host__ __device__ mat33 so3()
		{
			mat33 mat;
			mat.m00() = 0;
			mat.m01() = -xi.z;
			mat.m02() = xi.y;
			mat.m10() = xi.z;
			mat.m11() = 0;
			mat.m12() = -xi.x;
			mat.m20() = -xi.y;
			mat.m21() = xi.x;
			mat.m22() = 0;
			return mat;
		};

		float3 xi;
	};
}
