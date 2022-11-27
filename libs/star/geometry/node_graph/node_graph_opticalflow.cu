#include <device_launch_parameters.h>
#include <star/math/vector_ops.hpp>
#include <star/visualization/Visualizer.h>
#include <star/common/common_texture_utils.h>
#include <star/geometry/node_graph/node_graph_opticalflow.h>
#include <star/geometry/constants.h>

namespace star::device
{
	// Atomic version: easy to implement
	// Node motion is at world coordinate
	// Ideal: use pefect camera points to compute motion
	__global__ void AtomicSumIdealNodeMotionFromOpticalFlowKernel(
		cudaTextureObject_t vertex_confid_map_prev,
		cudaTextureObject_t vertex_confid_map,
		cudaTextureObject_t rgbd_map_prev,
		cudaTextureObject_t rgbd_map,
		cudaTextureObject_t opticalflow_map, // (dx, dy) in pixel
		cudaTextureObject_t index_map_prev,	 // Surfel index
		const ushortX<d_surfel_knn_size> *__restrict__ surfel_knn,
		const floatX<d_surfel_knn_size> *__restrict__ surfel_knn_spatial_weight,
		float4 *__restrict__ node_motion_pred,
		const unsigned image_width,
		const unsigned image_height,
		mat34 extrinsic,
		Intrinsic intrinsic)
	{
		const auto idx_prev = threadIdx.x + blockDim.x * blockIdx.x;
		const auto idy_prev = threadIdx.y + blockDim.y * blockIdx.y;
		if (idx_prev > image_width || idy_prev >= image_height)
			return;

		float2 opticalflow = tex2D<float2>(opticalflow_map, idx_prev, idy_prev);
		// TODO: add an optical flow: pixel to distance
		float idx = float(idx_prev) + opticalflow.x;
		float idy = float(idy_prev) + opticalflow.y;
		float4 rgbd = tex2D<float4>(rgbd_map, idx, idy); // TODO: change to bi-linear interpolation

		unsigned index_prev = tex2D<unsigned>(index_map_prev, idx_prev, idy_prev); // corresponding surfel index
		float4 rgbd_prev = tex2D<float4>(rgbd_map_prev, idx_prev, idy_prev);
		if (index_prev == 0xFFFFFFFF || rgbd.w == 0.f)
			return;
		unsigned short nearest_nid = surfel_knn[index_prev][0];
		float spatial_weight = surfel_knn_spatial_weight[index_prev][0];
		// The 3D motion in world
		float depth = 1.f / rgbd.w;
		float depth_prev = 1.f / rgbd_prev.w;

		// The vertex corresponding to depth
		float3 vertex = make_float3(
			(idx - intrinsic.principal_x) / intrinsic.focal_x * depth,
			(idy - intrinsic.principal_y) / intrinsic.focal_y * depth,
			depth);
		float3 vertex_prev = make_float3(
			(idx_prev - intrinsic.principal_x) / intrinsic.focal_x * depth_prev,
			(idy_prev - intrinsic.principal_y) / intrinsic.focal_y * depth_prev,
			depth_prev);

		float3 est_motion_cam = make_float3(
			vertex.x - vertex_prev.x,
			vertex.y - vertex_prev.y,
			vertex.z - vertex_prev.z);
		// Filter out wrong match
		if (fabs(est_motion_cam.z) > 0.05f) // 5cm
			return;
		est_motion_cam *= spatial_weight;
		float3 est_motion_world = extrinsic.rot * est_motion_cam;
		// Atomic add
		atomicAdd(&(node_motion_pred[nearest_nid].x), est_motion_world.x);
		atomicAdd(&(node_motion_pred[nearest_nid].y), est_motion_world.y);
		atomicAdd(&(node_motion_pred[nearest_nid].z), est_motion_world.z);
		atomicAdd(&(node_motion_pred[nearest_nid].w), spatial_weight);
	}

	__global__ void AverageNodeMotionKernel(
		float4 *__restrict__ node_motion_pred,
		const unsigned node_size)
	{
		const auto idx = threadIdx.x + blockDim.x * blockIdx.x;
		if (idx > node_size)
			return;

		float4 _node_motion_pred = node_motion_pred[idx];
		if (_node_motion_pred.w == 0.f)
			return;
		_node_motion_pred *= (1.f / _node_motion_pred.w);
		node_motion_pred[idx] = _node_motion_pred;
	}

	// Ideal: use pefect camera points to compute motion
	__global__ void IdealSurfelMotionFromOpticalFlowKernel(
		cudaTextureObject_t vertex_confid_map_prev,
		cudaTextureObject_t vertex_confid_map,
		cudaTextureObject_t rgbd_map_prev,
		cudaTextureObject_t rgbd_map,
		cudaTextureObject_t opticalflow_map, // (dx, dy) in pixel
		cudaTextureObject_t index_map_prev,	 // Surfel index
		float4 *__restrict__ surfel_motion_pred,
		const unsigned image_width,
		const unsigned image_height,
		mat34 extrinsic,
		Intrinsic intrinsic)
	{
		const auto idx_prev = threadIdx.x + blockDim.x * blockIdx.x;
		const auto idy_prev = threadIdx.y + blockDim.y * blockIdx.y;
		if (idx_prev > image_width || idy_prev >= image_height)
			return;

		float2 opticalflow = tex2D<float2>(opticalflow_map, idx_prev, idy_prev);

		// TODO: add an optical flow: pixel to distance
		float idx = float(idx_prev) + opticalflow.x;
		float idy = float(idy_prev) + opticalflow.y;
		unsigned rn_idx = __float2uint_rn(idx);
		unsigned rn_idy = __float2uint_rn(idy);
		float4 rgbd = tex2D<float4>(rgbd_map, rn_idx, rn_idy);					   // TODO: change to bi-linear interpolation
		unsigned index_prev = tex2D<unsigned>(index_map_prev, idx_prev, idy_prev); // corresponding surfel index
		float4 rgbd_prev = tex2D<float4>(rgbd_map_prev, idx_prev, idy_prev);
		if (index_prev == 0xFFFFFFFF || rgbd.w == 0.f)
			return;
		// The 3D motion in world
		float depth = 1.f / rgbd.w;
		float depth_prev = 1.f / rgbd_prev.w;

		// The vertex corresponding to depth
		float3 vertex = make_float3(
			(idx - intrinsic.principal_x) / intrinsic.focal_x * depth,
			(idy - intrinsic.principal_y) / intrinsic.focal_y * depth,
			depth);
		float3 vertex_prev = make_float3(
			(idx_prev - intrinsic.principal_x) / intrinsic.focal_x * depth_prev,
			(idy_prev - intrinsic.principal_y) / intrinsic.focal_y * depth_prev,
			depth_prev);

		float3 est_motion_cam = make_float3(
			vertex.x - vertex_prev.x,
			vertex.y - vertex_prev.y,
			vertex.z - vertex_prev.z);
		if (fabs(est_motion_cam.z) > 0.10f) // Z-shift in 10cm
			return;

		float3 est_motion_world = extrinsic.rot * est_motion_cam;
		surfel_motion_pred[index_prev] = make_float4(
			est_motion_world.x, est_motion_world.y, est_motion_world.z, 1.f);
	}

	// Just check what raw-opticalflow looks like in 3D
	__global__ void SurfelMotion2DFromOpticalFlowKernel(
		cudaTextureObject_t rgbd_map_prev,
		cudaTextureObject_t opticalflow_map, // (dx, dy) in pixel
		cudaTextureObject_t index_map_prev,	 // Surfel index
		float4 *__restrict__ surfel_motion_pred,
		const unsigned image_width,
		const unsigned image_height,
		mat34 extrinsic,
		Intrinsic intrinsic)
	{
		const auto idx_prev = threadIdx.x + blockDim.x * blockIdx.x;
		const auto idy_prev = threadIdx.y + blockDim.y * blockIdx.y;
		if (idx_prev > image_width || idy_prev >= image_height)
			return;

		float2 opticalflow = tex2D<float2>(opticalflow_map, idx_prev, idy_prev);

		// TODO: add an optical flow: pixel to distance
		unsigned index_prev = tex2D<unsigned>(index_map_prev, idx_prev, idy_prev); // corresponding surfel index
		float4 rgbd_prev = tex2D<float4>(rgbd_map_prev, idx_prev, idy_prev);
		if (index_prev == 0xFFFFFFFF)
			return;
		// The 3D motion in world
		float depth_prev = 1.f / rgbd_prev.w;

		// Opticalflow in 3D
		float3 of_3d = make_float3(
			(opticalflow.x) / intrinsic.focal_x * depth_prev,
			(opticalflow.y) / intrinsic.focal_y * depth_prev,
			0.f);

		float3 est_motion_world = extrinsic.rot * of_3d;
		surfel_motion_pred[index_prev] = make_float4(
			est_motion_world.x, est_motion_world.y, est_motion_world.z, 1.f);
	}
}

void star::AccumlateNodeMotionFromOpticalFlow(
	cudaTextureObject_t vertex_confid_map_prev, // Used for debug
	cudaTextureObject_t vertex_confid_map,
	cudaTextureObject_t rgbd_map_prev,
	cudaTextureObject_t rgbd_map,
	cudaTextureObject_t opticalflow_map, // (dx, dy) in pixel, defined on prev
	cudaTextureObject_t index_map_prev,	 // Surfel index
	GArrayView<ushortX<d_surfel_knn_size>> surfel_knn,
	GArrayView<floatX<d_surfel_knn_size>> surfel_knn_spatial_weight,
	GArraySlice<float4> node_motion_pred,
	const Extrinsic &extrinsic,
	const Intrinsic &intrinsic,
	cudaStream_t stream)
{
	unsigned image_width, image_height;
	query2DTextureExtent(rgbd_map, image_width, image_height);
	dim3 blk(16, 16);
	dim3 grid(divUp(image_width, blk.x), divUp(image_height, blk.y));
	// Sum
	device::AtomicSumIdealNodeMotionFromOpticalFlowKernel<<<grid, blk, 0, stream>>>(
		vertex_confid_map_prev,
		vertex_confid_map,
		rgbd_map_prev,
		rgbd_map,
		opticalflow_map, // (dx, dy) in pixel
		index_map_prev,	 // Surfel index
		surfel_knn.Ptr(),
		surfel_knn_spatial_weight.Ptr(),
		node_motion_pred.Ptr(),
		image_width,
		image_height,
		mat34(extrinsic),
		intrinsic);

	// Sync and check error
#if defined(CUDA_DEBUG_SYNC_CHECK)
	cudaSafeCall(cudaStreamSynchronize(stream));
	cudaSafeCall(cudaGetLastError());
#endif
}

void star::ResetNodeMotion(GArraySlice<float4> node_motion_pred, cudaStream_t stream)
{
	cudaSafeCall(cudaMemsetAsync(node_motion_pred.Ptr(), 0, sizeof(float4) * node_motion_pred.Size(), stream));
}

void star::AverageNodeMotion(GArraySlice<float4> node_motion_pred, cudaStream_t stream)
{
	unsigned node_size = node_motion_pred.Size();
	dim3 blk(128);
	dim3 grid(divUp(node_size, blk.x));
	device::AverageNodeMotionKernel<<<grid, blk, 0, stream>>>(
		node_motion_pred,
		node_size);

	// Sync and check error
#if defined(CUDA_DEBUG_SYNC_CHECK)
	cudaSafeCall(cudaStreamSynchronize(stream));
	cudaSafeCall(cudaGetLastError());
#endif
}

void star::EstimateSurfelMotionFromOpticalFlow(
	cudaTextureObject_t vertex_confid_map_prev, // Used for debug
	cudaTextureObject_t vertex_confid_map,
	cudaTextureObject_t rgbd_map_prev,
	cudaTextureObject_t rgbd_map,
	cudaTextureObject_t opticalflow_map, // (dx, dy) in pixel, defined on prev
	cudaTextureObject_t index_map_prev,	 // Surfel index
	GArraySlice<float4> surfel_motion_pred,
	const Extrinsic &extrinsic,
	const Intrinsic &intrinsic,
	cudaStream_t stream)
{
	unsigned image_width, image_height;
	query2DTextureExtent(rgbd_map, image_width, image_height);
	dim3 blk(16, 16);
	dim3 grid(divUp(image_width, blk.x), divUp(image_height, blk.y));

	// Reset
	cudaSafeCall(cudaMemsetAsync(surfel_motion_pred.Ptr(), 0, sizeof(float4) * surfel_motion_pred.Size(), stream));
	// Compute
	device::IdealSurfelMotionFromOpticalFlowKernel<<<grid, blk, 0, stream>>>(
		vertex_confid_map_prev,
		vertex_confid_map,
		rgbd_map_prev,
		rgbd_map,
		opticalflow_map, // (dx, dy) in pixel
		index_map_prev,	 // Surfel index
		surfel_motion_pred.Ptr(),
		image_width,
		image_height,
		mat34(extrinsic),
		intrinsic);

	// Sync and check error
#if defined(CUDA_DEBUG_SYNC_CHECK)
	cudaSafeCall(cudaStreamSynchronize(stream));
	cudaSafeCall(cudaGetLastError());
#endif
}

void star::SurfelMotion2DFromOpticalFlow(
	cudaTextureObject_t rgbd_map_prev,
	cudaTextureObject_t opticalflow_map, // (dx, dy) in pixel, defined on prev
	cudaTextureObject_t index_map_prev,	 // Surfel index
	GArraySlice<float4> surfel_motion_pred,
	const Extrinsic &extrinsic,
	const Intrinsic &intrinsic,
	cudaStream_t stream)
{
	unsigned image_width, image_height;
	query2DTextureExtent(rgbd_map_prev, image_width, image_height);
	dim3 blk(16, 16);
	dim3 grid(divUp(image_width, blk.x), divUp(image_height, blk.y));

	// Reset
	cudaSafeCall(cudaMemsetAsync(surfel_motion_pred.Ptr(), 0, sizeof(float4) * surfel_motion_pred.Size(), stream));
	// Compute
	device::SurfelMotion2DFromOpticalFlowKernel<<<grid, blk, 0, stream>>>(
		rgbd_map_prev,
		opticalflow_map, // (dx, dy) in pixel
		index_map_prev,	 // Surfel index
		surfel_motion_pred.Ptr(),
		image_width,
		image_height,
		mat34(extrinsic),
		intrinsic);

	// Sync and check error
#if defined(CUDA_DEBUG_SYNC_CHECK)
	cudaSafeCall(cudaStreamSynchronize(stream));
	cudaSafeCall(cudaGetLastError());
#endif
}