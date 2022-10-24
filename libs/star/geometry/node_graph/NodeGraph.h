#pragma once
#include <memory>
#include <cuda_fp16.h>
#include "common/ArrayView.h"
#include "common/GBufferArray.h"
#include "common/SyncArray.h"
#include "common/macro_utils.h"
#include "math/DualQuaternion.hpp"
#include "star/types/solver_types.h"
#include "star/types/skinner_types.h"
#include "star/common/ConfigParser.h"
#include "star/WarpField.h"
#include "star/geometry/VoxelSubsampler.h"
#include "star/geometry/node_graph/node_graph_constants.h"
#include "star/geometry/VoxelSubsamplerSorting.h"
#include "visualization/Visualizer.h"

// Macros
#define DELETED_NODE_STATUS 0xffffffff
#define ACTIVE_NODE_STATUS 0

namespace star {

	class NodeGraph {
	public:
		using Ptr = std::shared_ptr<NodeGraph>;
		using ConstPtr = std::shared_ptr<NodeGraph const>;
		NodeGraph(const float node_radius);
		~NodeGraph();
		STAR_NO_COPY_ASSIGN_MOVE(NodeGraph);
	
		/* Initialization
		*/
		void InitializeNodeGraphFromVertex(
			const GArrayView<float4>& vertex,
            const unsigned current_time,
			const bool use_ref,
			cudaStream_t stream = 0
		);
		void InitializeNodeGraphFromNode(  // Device method
			const GArrayView<float4>& node_coords,
            const unsigned current_time,
			const bool use_ref,
			cudaStream_t stream = 0
		);
		void InitializeNodeGraphFromNode(  // Host method
			const std::vector<float4>& node_coords,
			const unsigned current_time,
			const bool use_ref,
			cudaStream_t stream = 0
		);
		/* Expansion
		*/
		void NaiveExpandNodeGraphFromVertexUnsupported(
			const GArrayView<float4>& vertex_unsupported,
			const unsigned current_time,
			cudaStream_t stream
		);
		/* Update after removal; Default update
		*/
		void BuildNodeGraphFromScratch(const bool use_ref, cudaStream_t stream);  // Building
		void BuildNodeGraphFromDistance(cudaStream_t stream);
		void UpdateNodeDistance(const bool use_ref, cudaStream_t stream);
		void ResetNodeGraphConnection(cudaStream_t stream);
		void ComputeNodeGraphConnectionFromSemantic(const floatX<d_max_num_semantic>& dynamic_regulation, cudaStream_t stream);

		// Access API
		GArrayView<float4> GetReferenceNodeCoordinate() const {
			return m_reference_node_coords.DeviceArrayReadOnly();
		}
		GArrayView<float4> GetLiveNodeCoordinate() const {
			return m_live_node_coords.DeviceArrayReadOnly();
		}
		GArrayView<ushortX<d_node_knn_size>> GetNodeKnn() const {
			return m_node_knn.DeviceArrayReadOnly();
		}
		GArrayView<floatX<d_node_knn_size>> GetNodeKnnConnectWeight() const {
			return m_node_knn_connect_weight.DeviceArrayReadOnly();
		}
		GArrayView<floatX<d_node_knn_size>> GetNodeKnnSpatialWeight() const {
			return m_node_knn_spatial_weight.DeviceArrayReadOnly();
		}
		unsigned GetNodeSize() const { return m_node_size; }
		unsigned GetPrevNodeSize() const { return m_prev_node_size; }
		NodeGraph4Solver GenerateNodeGraph4Solver() const;
		WarpField::DeformationAcess DeformAccess();
		float GetNodeRadiusSquare() const { return m_node_radius * m_node_radius; }
		GArraySlice<half> GetNodeDistance() { return GArraySlice<half>(m_node_distance); }
		GArraySlice<uint2> GetNodeStatus() { return m_node_status.Slice(); }
		GArrayView<uint2> GetNodeStatusReadOnly() { return m_node_status.View(); }
		GArraySlice<unsigned> GetCounterNodeOuttrack() { return m_counter_node_outtrack.Slice(); }
		GArrayView<unsigned> GetCounterNodeOuttrackReadOnly() const { return m_counter_node_outtrack.View(); }
		GArrayView<unsigned> GetNodeInitialTime() { return m_node_initial_time.View(); }
		GArraySlice<ucharX<d_max_num_semantic>> GetNodeSemanticProb() { return m_node_semantic_prob.Slice(); }
		GArrayView<ucharX<d_max_num_semantic>> GetNodeSemanticProbReadOnly() const { return m_node_semantic_prob.View(); }
		GArraySlice<float> GetNodeSemanticProbVoteBuffer() { return m_node_semantic_prob_vote_buffer.Slice(); }
	private:
		void updateNodeCoordinate(const GArrayView<float4>& node_coords, const unsigned current_time, cudaStream_t stream = 0);
		void updateNodeCoordinate(const std::vector<float4>& node_coords, const unsigned current_time, cudaStream_t stream = 0);
		void updateAppendNodeInitialTime(
			const unsigned current_time, const unsigned prev_node_size, const unsigned append_node_size, cudaStream_t stream = 0);
		void buildNodeGraphPair(cudaStream_t stream = 0);
		void resizeNodeSize(unsigned node_size);
		
		// Sampling
		SyncArray<float4> m_node_vertex_candidate;  // Buffer
		VoxelSubsampler::Ptr m_vertex_subsampler;
		float m_node_radius;

		unsigned m_node_size;
		unsigned m_prev_node_size;
		unsigned* m_newly_remove_count;
		// Data (Keep): [Notice] Defaultly only sync on device
		SyncArray<float4> m_reference_node_coords;
		SyncArray<float4> m_live_node_coords;
		SyncArray<ushortX<d_node_knn_size>> m_node_knn;
		SyncArray<floatX<d_node_knn_size>> m_node_knn_spatial_weight;  // (0~1)
		SyncArray<floatX<d_node_knn_size>> m_node_knn_connect_weight;  // (0~1)
		GBufferArray<ushort3> m_node_graph_pair;		// (i, j, k): j is i's kth neighbor
		// Node attribute
		GBufferArray<ucharX<d_max_num_semantic>> m_node_semantic_prob;
		GBufferArray<float> m_node_semantic_prob_vote_buffer;
		// Auxilary
		GBufferArray<uint2> m_node_status;  // (compression, frozen_time)
		GBufferArray<unsigned> m_counter_node_outtrack;
		GBufferArray<unsigned> m_node_initial_time;    // At which time, new nodes are appended
		GArray<half> m_node_distance;		 // Use this to maintain inner distance

	public:
		NodeGraph4Skinner GenerateNodeGraph4Skinner() const;
		// Set tar's ref from src's live. "ReAnchor" the ref geometry
		static void ReAnchor(
			NodeGraph::ConstPtr src_node_graph,
			NodeGraph::Ptr tar_node_graph,
			cudaStream_t stream
		);

		/* Utility
		*/
		void AppendNodeFromVertexHost(
			const GArrayView<float4>& vertex,
			std::vector<float4>& reference_node,
			std::vector<float4>& live_node,
			cudaStream_t stream
		);

	};
}
