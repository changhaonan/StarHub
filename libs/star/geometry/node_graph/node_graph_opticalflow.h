#pragma once
#include <star/common/macro_utils.h>
#include <star/common/common_types.h>
#include <star/common/ArrayView.h>
#include <star/common/ArraySlice.h>
#include <star/math/DualQuaternion.hpp>
#include <star/geometry/constants.h>

namespace star
{
	void AccumlateNodeMotionFromOpticalFlow(
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
		cudaStream_t stream);

	void ResetNodeMotion(GArraySlice<float4> node_motion_pred, cudaStream_t stream);
	void AverageNodeMotion(GArraySlice<float4> node_motion_pred, cudaStream_t stream);

	// Debug method
	void EstimateSurfelMotionFromOpticalFlow(
		cudaTextureObject_t vertex_confid_map_prev, // Used for debug
		cudaTextureObject_t vertex_confid_map,
		cudaTextureObject_t rgbd_map_prev,
		cudaTextureObject_t rgbd_map,
		cudaTextureObject_t opticalflow_map, // (dx, dy) in pixel, defined on prev
		cudaTextureObject_t index_map_prev,	 // Surfel index
		GArraySlice<float4> surfel_motion_pred,
		const Extrinsic &extrinsic,
		const Intrinsic &intrinsic,
		cudaStream_t stream);

	void SurfelMotion2DFromOpticalFlow(
		cudaTextureObject_t rgbd_map_prev,
		cudaTextureObject_t opticalflow_map, // (dx, dy) in pixel, defined on prev
		cudaTextureObject_t index_map_prev,	 // Surfel index
		GArraySlice<float4> surfel_motion_pred,
		const Extrinsic &extrinsic,
		const Intrinsic &intrinsic,
		cudaStream_t stream);
}