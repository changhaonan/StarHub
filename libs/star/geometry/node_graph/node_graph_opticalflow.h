/**
 * @author Haonan Chang
 * @email chnme40cs@gmail.com
 * @create date 2022-04-20
 * @modify date 2022-04-20
 * @desc Estimate the node motion from optical flow & surfel skinning
 */
#pragma once

#include "common/macro_utils.h"
#include "common/common_types.h"
#include "star/common/global_configs.h"
#include "common/ArrayView.h"
#include "common/ArraySlice.h"
#include "math/DualQuaternion.hpp"

namespace star {

	void AccumlateNodeMotionFromOpticalFlow(
		cudaTextureObject_t vertex_confid_map_prev,  // Used for debug
		cudaTextureObject_t vertex_confid_map,
		cudaTextureObject_t rgbd_map_prev,
		cudaTextureObject_t rgbd_map,
		cudaTextureObject_t opticalflow_map,  // (dx, dy) in pixel, defined on prev
		cudaTextureObject_t index_map_prev,  // Surfel index
		GArrayView<ushortX<d_surfel_knn_size>> surfel_knn,
		GArrayView<floatX<d_surfel_knn_size>> surfel_knn_spatial_weight,
		GArraySlice<float4> node_motion_pred,
		const Extrinsic& extrinsic,
		const Intrinsic& intrinsic,
		cudaStream_t stream
	);
	
	void ResetNodeMotion(GArraySlice<float4> node_motion_pred, cudaStream_t stream);
	void AverageNodeMotion(GArraySlice<float4> node_motion_pred, cudaStream_t stream);

	// Debug method
	void EstimateSurfelMotionFromOpticalFlow(
		cudaTextureObject_t vertex_confid_map_prev,  // Used for debug
		cudaTextureObject_t vertex_confid_map,
		cudaTextureObject_t rgbd_map_prev,
		cudaTextureObject_t rgbd_map,
		cudaTextureObject_t opticalflow_map,  // (dx, dy) in pixel, defined on prev
		cudaTextureObject_t index_map_prev,  // Surfel index
		GArraySlice<float4> surfel_motion_pred,
		const Extrinsic& extrinsic,
		const Intrinsic& intrinsic,
		cudaStream_t stream
	);
}