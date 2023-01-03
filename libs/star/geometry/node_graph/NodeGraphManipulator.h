#pragma once
#include <cuda_fp16.h>
#include <star/common/common_types.h>
#include <star/common/ArrayView.h>
#include <star/common/ArraySlice.h>
#include <star/common/SyncArray.h>
#include <star/geometry/constants.h>
#include <star/geometry/node_graph/NodeGraph.h>

namespace star
{
	/** \brief Provide high-level manipulation towards node graph
	 */
	class NodeGraphManipulator
	{
	public:
		// Given new vertex, compute new head
		static void CheckSurfelCandidateSupportStatus(
			const GArrayView<float4> &vertex_confid_candidate,
			const GArrayView<float4> &node_coord,
			const GArrayView<uint2> &node_status,
			GArraySlice<unsigned> candidate_validity_indicator,
			GArraySlice<unsigned> candidate_unsupported_indicator,
			GArraySlice<ushortX<d_surfel_knn_size>> candidate_knn,
			cudaStream_t stream,
			const float node_radius_square);

		// Remove those node that are out of track
		static void UpdateCounterNodeOutTrack(
			const GArrayView<unsigned> &surfel_validity,
			const GArrayView<ushortX<d_surfel_knn_size>> &surfel_knn,
			GArraySlice<unsigned> counter_node_outtrack,
			cudaStream_t stream);
		static void RemoveNodeOutTrackSync(
			const GArrayView<ushortX<d_surfel_knn_size>> &node_knn,
			const GArrayView<unsigned> &counter_node_outtrack,
			GArraySlice<uint2> node_status,
			GArraySlice<half> node_distance,
			unsigned &num_node_remove_count,
			const float counter_node_outtrack_threshold,
			const unsigned frozen_time,
			cudaStream_t stream);

		// Update the semantic state for node
		static void UpdateNodeSemanticProb(
			const GArrayView<ushortX<d_surfel_knn_size>> &surfel_knn,
			const GArrayView<ucharX<d_max_num_semantic>> &surfel_semantic_prob,
			GArraySlice<ucharX<d_max_num_semantic>> node_semantic_prob,
			GArraySlice<float> node_semantic_prob_vote_buffer,
			cudaStream_t stream);

		// Update the new semantic state for node
		static void UpdateIncNodeSemanticProb(
			const GArrayView<ushortX<d_surfel_knn_size>> &surfel_knn,
			const GArrayView<ucharX<d_max_num_semantic>> &surfel_semantic_prob,
			GArraySlice<ucharX<d_max_num_semantic>> node_semantic_prob,
			GArraySlice<float> node_semantic_prob_vote_buffer,
			unsigned num_prev_node,
			cudaStream_t stream);

		// Utility function
		// Compute the average node movement for the choose node
		// Metric method
		static void AvergeNodeMovementAndPos(
			const GArrayView<float4> &node_coord,
			const GArrayView<DualQuaternion> &delta_node_deform,
			const GArrayView<unsigned short> &node_list,
			GArraySlice<DualQuaternion> node_deform,
			DualQuaternion &average_node_se3,
			float3 &average_node_pos,
			cudaStream_t stream);

		// Select by semantic
		static void SelectNodeBySemanticAtomic(
			const GArrayView<ucharX<d_max_num_semantic>> &node_semantic_prob,
			const unsigned short semantic_id,
			GArraySlice<unsigned short> node_list_selected,
			unsigned& num_node_selected,
			cudaStream_t stream);
	};
}
