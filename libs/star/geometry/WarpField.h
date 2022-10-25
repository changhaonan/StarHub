#pragma once

#include <star/geometry/constants.h>
#include <star/common/sanity_check.h>
#include <star/common/ArrayView.h>
#include <star/math/DualQuaternion.hpp>

namespace star
{

	// Forware declaration
	class WarpFieldUpdater;
	class SurfelNodeDeformer;
	class SurfelGeometryUnifiedUpdater;

	/*
	 * \brief: WarpField is currently serving as an interface class. No data
	 * will be saved here. It is working as an interface bridging Node Graph
	 * with others.
	 */
	class WarpField
	{
	public:
		// Access type
		struct SolverInput
		{
			GArrayView<DualQuaternion> node_se3;
			GArrayView<float4> reference_node_coords;
			GArrayView<ushort2> node_graph;
			GArrayView<float> node_error; // Node alignment error
		};
		struct LiveGeometryUpdaterInput
		{
			GArrayView<float4> live_node_coords;
			GArrayView<float4> reference_node_coords;
			GArrayView<DualQuaternion> node_se3;
		};
		struct SkinnerInput
		{
			GArrayView<float4> reference_node_coords;
			GArrayView<float4> live_node_coords; // For from live
			GArrayView<ushort4> node_knn_low;
			GArrayView<ushort4> node_knn_high;
			GArrayView<float4> node_knn_weight;
		};

		struct DeformationAcess
		{
			GArraySlice<float4> reference_node_coords;
			GArraySlice<float4> live_node_coords;
			GArrayView<ushortX<d_node_knn_size>> node_knn;
			GArrayView<floatX<d_node_knn_size>> node_knn_spatial_weight;
			GArrayView<floatX<d_node_knn_size>> node_knn_connect_weight;
		};
	};

}