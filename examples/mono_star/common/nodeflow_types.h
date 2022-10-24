#pragma once
#include "Constants.h"
#include "frame_buffer.h"
#include "solver_types.h"

namespace star {

	struct NodeFlowBuffer : public FrameBuffer {
		NodeFlowBuffer() { create(); }
		~NodeFlowBuffer() { release(); }

#define DEBUG_SURFEL_MOTION	
		void create() override {
			node_motion_pred.AllocateBuffer(Constants::kMaxNumNodes);
#ifdef DEBUG_SURFEL_MOTION
			surfel_motion_pred.AllocateBuffer(Constants::kMaxNumSurfels);
#endif
		}
		void release() override {
			node_motion_pred.ReleaseBuffer();
#ifdef DEBUG_SURFEL_MOTION
			surfel_motion_pred.ReleaseBuffer();
#endif
		}

		GBufferArray<float4>& NodeFlow() { return node_motion_pred; }
		GArrayView<float4> NodeFlowReadOnly() const { return node_motion_pred.View(); }
		GBufferArray<float4> node_motion_pred;

		void log(const unsigned buffer_idx) override {
			std::string text = stringFormat(" NodeFlow - frame %d - buffer: %d ", frame_idx, buffer_idx);
			std::string aligned_text = stringAlign2Center(text, logging_length, "=");
			std::cout << aligned_text << std::endl;
		};

#ifdef DEBUG_SURFEL_MOTION
		GBufferArray<float4>& SurfelFlow() { return surfel_motion_pred; }
		GArrayView<float4> SurfelFlowReadOnly() const { return surfel_motion_pred.View(); }
		GBufferArray<float4> surfel_motion_pred;
#endif

		// Solver-API
		NodeFlow4Solver GenerateNodeFlow4Solver() const {
			NodeFlow4Solver nodeflow4solver;
			nodeflow4solver.node_motion_pred = node_motion_pred.View();
			nodeflow4solver.num_node = node_motion_pred.ArraySize();
			return nodeflow4solver;
		}
	};

}