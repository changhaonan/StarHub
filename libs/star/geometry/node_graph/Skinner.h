#pragma once
#include "star/types/skinner_types.h"
#include "star/geometry/node_graph/NodeGraph.h"

namespace star {
	class Skinner {
	public:
		// Full skinner
		static void PerformSkinningFromRef(
			Geometry4Skinner& geometry, NodeGraph4Skinner& node_graph, cudaStream_t stream = 0);
		static void PerformSkinningFromLive(
			Geometry4Skinner& geometry, NodeGraph4Skinner& node_graph, cudaStream_t stream = 0);
		static void PerformSkinningFromLiveWithSemantic(
			Geometry4Skinner& geometry, NodeGraph4Skinner& node_graph, cudaStream_t stream
		);

		// Increment skinner
		// Skinning all surfel between newly added node & newly added surfel
		static void PerformIncSkinnningFromLive(
			Geometry4Skinner& geometry, NodeGraph4Skinner& node_graph, 
			const unsigned num_prev_node_size, const unsigned num_remaining_surfel, cudaStream_t stream
		);

		// Update the surfel connection for all surfel
		static void UpdateSkinnningConnection(
			Geometry4Skinner& geometry, NodeGraph4Skinner& node_graph, cudaStream_t stream
		);
	};
}