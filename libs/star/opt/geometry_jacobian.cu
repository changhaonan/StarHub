#include <star/opt/geometry_jacobian.h>

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
		const unsigned offset)
	{
		// Compute the residual terms
		residual[offset] = dot(target_normal, source_vertex - target_vertex);
		gradient.gradient[offset].rotation = cross(source_vertex, target_normal);
		gradient.gradient[offset].translation = target_normal;
	}

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
		const unsigned offset)
	{
		const float3 source_vertex_camera = world2camera.apply_se3(source_vertex);
		const float inv_depth_source_vertex = 1.f / source_vertex_camera.z;
		const float2 source_pixel = intrinsic * (source_vertex_camera * inv_depth_source_vertex);
		const float2 pixel_diff = source_pixel - target_pixel;
		if (norm(pixel_diff) < 0.1f)
		{
			// Ignore small offset
			residual[offset] = 0.f;
			residual[offset + 1] = 0.f;
			gradient.gradient[offset].translation = make_float3(0.f, 0.f, 0.f);
			gradient.gradient[offset + 1].translation = make_float3(0.f, 0.f, 0.f);
			gradient.gradient[offset].rotation = make_float3(0.f, 0.f, 0.f);
			gradient.gradient[offset + 1].rotation = make_float3(0.f, 0.f, 0.f);
		}
		else
		{
			residual[offset] = pixel_diff.x;
			residual[offset + 1] = pixel_diff.y;

			const mat23 intrinsic_from_world = (intrinsic * world2camera.rot) * inv_depth_source_vertex;
			gradient.gradient[offset].translation = intrinsic_from_world.row(0);
			gradient.gradient[offset + 1].translation = intrinsic_from_world.row(1);
			twist_rotation twist_source(-source_vertex);
			const mat23 gradient_rot = (intrinsic_from_world * twist_source.so3());
			gradient.gradient[offset].rotation = gradient_rot.row(0);
			gradient.gradient[offset + 1].rotation = gradient_rot.row(1);
		}
	}

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

	)
	{
		// 1. Residual
		const float3 diff_vertex = (source_vertex - target_vertex);
		const float2 e = project_rot_world2camera * diff_vertex;

		if (norm(e) < 0.01f)
		{
			// 2.1. Ignore small offset
			residual[offset] = 0.f;
			residual[offset + 1] = 0.f;
			gradient.gradient[offset].translation = make_float3(0.f, 0.f, 0.f);
			gradient.gradient[offset + 1].translation = make_float3(0.f, 0.f, 0.f);
			gradient.gradient[offset].rotation = make_float3(0.f, 0.f, 0.f);
			gradient.gradient[offset + 1].rotation = make_float3(0.f, 0.f, 0.f);
		}
		else
		{
			residual[offset] = e.x;
			residual[offset + 1] = e.y;
			// 2.1. J_t
			gradient.gradient[offset].translation = project_rot_world2camera.row(0);
			gradient.gradient[offset + 1].translation = project_rot_world2camera.row(1);
			// 2.2. J_r
			twist_rotation twist_source(-source_vertex);
			const mat23 gradient_rot = (project_rot_world2camera * twist_source.so3());
			gradient.gradient[offset].rotation = gradient_rot.row(0);
			gradient.gradient[offset + 1].rotation = gradient_rot.row(1);
		}
	}

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
		floatX<d_dense_image_residual_dim> &residual)
	{
		// PICP starts at 0
		computePointToPlanceICPTermJacobianResidual(
			target_vertex,
			target_normal,
			source_vertex,
			gradient,
			residual,
			0);
		// OpticalFlow starts at 1
		mat23 P;
		P.m00() = 1.f;
		P.m01() = 0.f;
		P.m02() = 0.f;
		P.m00() = 0.f;
		P.m01() = 1.f;
		P.m02() = 0.f;
		mat23 project_rot_world2camera = P * world2camera.rot;
		computeModifiedOpticalFlow2DTermJacobianResidual(
			target_vertex,
			source_vertex,
			project_rot_world2camera,
			// The output
			gradient,
			residual,
			1);
	}
}