#pragma once
#include <star/common/containers/device_array.hpp>
#include <star/common/containers/kernel_containers.hpp>
#include <star/common/containers/safe_call.cuh>
#include <star/common/types/typeX.h>

// Do not use cuda on Eigen
#ifndef EIGEN_NO_CUDA
#define EIGEN_NO_CUDA
#endif
#include <Eigen/Eigen>

// Cuda types
#include <vector_functions.h>
#include <vector>

namespace star
{

	// Access of common Eigen types
	using Matrix3f = Eigen::Matrix3f;
	using Vector3f = Eigen::Vector3f;
	using Matrix4f = Eigen::Matrix4f;
	using Vector4f = Eigen::Vector4f;
	using Matrix6f = Eigen::Matrix<float, 6, 6>;
	using Vector6f = Eigen::Matrix<float, 6, 1>;
	using MatrixXf = Eigen::MatrixXf;
	using VectorXf = Eigen::VectorXf;
	using Isometry3f = Eigen::Isometry3f;

	/* Types for host and device accessed gpu containers
	 */
	template <typename T>
	using GArray = DeviceArrayPCL<T>;

	template <typename T>
	using GArray2D = DeviceArray2DPCL<T>;

	namespace device
	{
		// Types for device accessed gpu containers
		template <typename T>
		using DevicePtr = DevPtr<T>;

		template <typename T>
		using PtrSz = PtrSzPCL<T>;

		template <typename T>
		using PtrStep = PtrStepPCL<T>;

		template <typename T>
		using PtrStepSz = PtrStepSzPCL<T>;
	}

	// Extrinsic
	using Extrinsic = Eigen::Matrix4f;

	/* The intrinsic and inverse intrinsic parameters
	 */
	struct Intrinsic
	{
		// Allow construction on both host and device
		__host__ __device__ Intrinsic()
			: principal_x(0), principal_y(0),
			  focal_x(0), focal_y(0) {}

		__host__ __device__ Intrinsic(
			const Intrinsic &intrinsic) : principal_x(intrinsic.principal_x),
										  principal_y(intrinsic.principal_y),
										  focal_x(intrinsic.focal_x),
										  focal_y(intrinsic.focal_y) {}

		__host__ __device__ Intrinsic(
			const float focal_x_, const float focal_y_,
			const float principal_x_, const float principal_y_) : principal_x(principal_x_), principal_y(principal_y_),
																  focal_x(focal_x_), focal_y(focal_y_) {}

		// Cast to float4
		__host__ operator float4()
		{
			return make_float4(principal_x, principal_y, focal_x, focal_y);
		}

		// Sacle
		__host__ __device__ Intrinsic scale(const float scale_) const
		{
			return Intrinsic(
				focal_x * scale_, focal_y * scale_, principal_x * scale_, principal_y * scale_);
		}

		// The paramters for camera intrinsic
		float principal_x, principal_y;
		float focal_x, focal_y;
	};

	struct IntrinsicInverse
	{
		__host__ __device__ IntrinsicInverse()
			: principal_x(0), principal_y(0),
			  inv_focal_x(0), inv_focal_y(0) {}

		// The paramters for camera intrinsic
		float principal_x, principal_y;
		float inv_focal_x, inv_focal_y;
	};

	// The texture collection of a given array
	struct CudaTextureSurface
	{
		cudaTextureObject_t texture;
		cudaSurfaceObject_t surface;
		cudaArray_t d_array;
	};

	// The divUp function provided by pcl
	using pcl::gpu::divUp;

	// The frame peroid for many usages
	struct FramePeriods
	{
		std::vector<int> start_frames; // included
		std::vector<int> end_frames;   // included
		size_t size() const { return start_frames.size(); }
	};

}