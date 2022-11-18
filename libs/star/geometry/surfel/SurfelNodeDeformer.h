#pragma once
#include <star/common/macro_utils.h>
#include <star/geometry/surfel/SurfelGeometry.h>
#include <star/geometry/WarpField.h>
#include <memory>

namespace star
{

	/* The deformer will perform forward and inverse
	 * warping given the geometry and warp field. It has
	 * full accessed to the SurfelGeometry instance.
	 */
	class SurfelNodeDeformer
	{
	public:
		// The processing of forward warp, may
		// use a node se3 different from the on in warp field
		static void ForwardWarpSurfelsAndNodes(
			WarpField::DeformationAcess warp_field,
			SurfelGeometry &geometry,
			const GArrayView<DualQuaternion> &node_se3,
			cudaStream_t stream = 0);

		// Debug method
		static void GenerateRandomDeformation(
			GArraySlice<DualQuaternion> &node_se3,
			const float trans_scale,
			const float rot_scale,
			cudaStream_t stream);
		static void CheckSurfelGeometySize(const SurfelGeometry &geometry);
	};

}
