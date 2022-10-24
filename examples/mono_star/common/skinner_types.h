#pragma once
#include <star/common/macro_utils.h>
#include <star/common/common_types.h>
#include <star/common/ArrayView.h>
#include <star/common/ArraySlice.h>
#include <star/common/types/typeX.h>
#include "global_configs.h"

namespace star
{
	struct Geometry4Skinner
	{
		GArrayView<float4> reference_vertex_confid;
		GArrayView<float4> reference_normal_radius;
		GArrayView<float4> live_vertex_confid;
		GArrayView<float4> live_normal_radius;
		GArrayView<ucharX<d_max_num_semantic>> surfel_semantic_prob; // (Optional, but can be useful)
		GArraySlice<ushortX<d_surfel_knn_size>> surfel_knn;
		GArraySlice<floatX<d_surfel_knn_size>> surfel_knn_spatial_weight;
		GArraySlice<floatX<d_surfel_knn_size>> surfel_knn_connect_weight;
	};

	struct NodeGraph4Skinner
	{
		GArrayView<float4> reference_node_coords;
		GArrayView<float4> live_node_coords;
		GArrayView<ucharX<d_max_num_semantic>> node_semantic_prob; // (Optional, but can be useful)
		GArrayView<uint2> node_status;
		GArrayView<ushortX<d_node_knn_size>> node_knn;
		GArrayView<floatX<d_node_knn_size>> node_knn_spatial_weight;
		GArrayView<floatX<d_node_knn_size>> node_knn_connect_weight;
		float node_radius_square;
	};
}