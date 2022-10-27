#pragma once
#include <star/common/macro_utils.h>
#include <star/common/common_types.h>
#include <star/common/ArrayView.h>
#include <star/common/ArraySlice.h>
#include <star/common/logging.h>
#include <star/common/common_utils.h>
#include <star/math/DualQuaternion.hpp>
#include <ostream>

namespace star
{

	/* Input for solver, Measurement, 2D, image-level
	 */
	struct Measure4Solver
	{
		cudaTextureObject_t vertex_confid_map[d_max_cam];
		cudaTextureObject_t normal_radius_map[d_max_cam];
		cudaTextureObject_t index_map[d_max_cam];
		unsigned num_cam;
	};

	/* Input for solver, Render, 2D, image-level
	 */
	struct Render4Solver
	{
		cudaTextureObject_t reference_vertex_map[d_max_cam];
		cudaTextureObject_t reference_normal_map[d_max_cam];
		cudaTextureObject_t index_map[d_max_cam];
		unsigned num_cam;
	};

	// Solver map
	struct SolverMaps
	{
		using Ptr = std::shared_ptr<SolverMaps>;

		cudaTextureObject_t reference_vertex_map[d_max_cam] = {0};
		cudaTextureObject_t reference_normal_map[d_max_cam] = {0};
		cudaTextureObject_t index_map[d_max_cam] = {0};
		cudaTextureObject_t normalized_rgbd_map[d_max_cam] = {0};
	};

	/* Input for solver, Geometry, 1D, surfel-level
	 */
	struct Geometry4Solver
	{
		GArrayView<ushortX<d_surfel_knn_size>> surfel_knn;
		GArrayView<floatX<d_surfel_knn_size>> surfel_knn_spatial_weight;
		GArrayView<floatX<d_surfel_knn_size>> surfel_knn_connect_weight;
		unsigned num_vertex;
	};

	/* Input for solver, NodeGraph, 1D, node-level
	 */
	struct NodeGraph4Solver
	{
		// Used for reg term
		GArrayView<float4> reference_node_coords;
		GArrayView<ushort3> node_graph;
		GArrayView<floatX<d_node_knn_size>> node_knn_connect_weight;
		// Used for node motion term
		GArrayView<ushortX<d_surfel_knn_size>> nodel_knn;
		GArrayView<floatX<d_surfel_knn_size>> node_knn_spatial_weight;
		unsigned num_node;
		float node_radius_square;
	};

	/* Input for solver, NodeFlow, 1D, node-level
	 */
	struct NodeFlow4Solver
	{
		GArrayView<float4> node_motion_pred;
		unsigned num_node;
	};

	/* Input for solver, OpticalFlow, 2D, image-level
	 */
	struct OpticalFlow4Solver
	{
		cudaTextureObject_t opticalflow_map[d_max_cam];
		unsigned num_cam;
	};
}