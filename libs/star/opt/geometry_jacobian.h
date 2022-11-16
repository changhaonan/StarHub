#pragma once
#include <star/common/common_types.h>
#include <star/math/vector_ops.hpp>
#include <star/math/DualQuaternion.hpp>
#include <star/math/twist.hpp>
#include <star/opt/solver_types.h>
#include <star/opt/constants.h>
#include <star/opt/huber_weight.h>

namespace star::device
{
	/**
	 * Plane2Plane ICP loss, all vertex, normal are defined in the same coordinate
	 * (e.g. camera or world).
	 * Note: only make sense, when error is small.
	 */
	__device__ void computePointToPlanceICPTermJacobianResidual(
		const float3 &target_vertex,
		const float3 &target_normal,
		const float3 &source_vertex,
		// The output
		GradientOfDenseImage &gradient,
		floatX<d_dense_image_residual_dim> &residual,
		const unsigned offset);

	/**
	 * OpticalFlow 2D loss:
	 * The source_vertex is supposed to projected to target_pixel.
	 * The 2D loss is defined on 2D distance.
	 * Note: only make sense, when error is small.
	 */
	__device__ void computeOpticalFlow2DTermJacobianResidual(
		const float2 &target_pixel,
		const float3 &source_vertex,
		const mat34 &world2camera,
		const mat23 &intrinsic,
		// The output
		GradientOfDenseImage &gradient,
		floatX<d_dense_image_residual_dim> &residual,
		const unsigned offset);

	/**
	 * Modified OpticalFlow 2D loss:
	 * Instead of computing pixel difference, compute the difference of vertex within the camera plane
	 * Note: only make sense, when error is small.
	 */
	__device__ void computeModifiedOpticalFlow2DTermJacobianResidual(
		const float3 &target_vertex,
		const float3 &source_vertex,
		const mat23 &project_rot_world2camera,
		// The output
		GradientOfDenseImage &gradient,
		floatX<d_dense_image_residual_dim> &residual,
		const unsigned offset

	);

	/**
	 * Picp & OpticalFlow loss:
	 * Hybrid loss of all mentioned above.
	 * Weight2D is the relative weight of opticalflow to picp (default as 1)
	 * source, target are all defined in world coordinate
	 */
	__device__ void computeDenseImageTermJacobianResidual(
		const float3 &target_vertex,
		const float3 &target_normal,
		const float2 &target_pixel,
		const float3 &source_vertex,
		const mat34 &world2camera,
		const mat23 &intrinsic,
		// The output
		GradientOfDenseImage &gradient,
		floatX<d_dense_image_residual_dim> &residual);

}