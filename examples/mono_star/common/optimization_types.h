#pragma once
#include <star/math/DualQuaternion.hpp>
#include "frame_buffer.h"

namespace star {
	
	struct OptimizationBuffer : public FrameBuffer {

		OptimizationBuffer() { create(); }
		~OptimizationBuffer() { release(); }
        void create() override {
			auto& config = ConfigParser::Instance();
			solved_node_se3.AllocateBuffer(Constants::kMaxNumNodes);
			num_cam = config.num_cam();
			for (auto cam_idx = 0; cam_idx < num_cam; ++cam_idx) {
				potential_pixel_pair[cam_idx].AllocateBuffer(
					size_t(config.downsample_img_cols(cam_idx)) * 
					size_t(config.downsample_img_rows(cam_idx))
				);
			}
		}
        void release() override {
			solved_node_se3.ReleaseBuffer();
			for (auto cam_idx = 0; cam_idx < num_cam; ++cam_idx) {
				potential_pixel_pair[cam_idx].ReleaseBuffer();
			}
		}

		// Fetch API
		GArraySlice<DualQuaternion> SolvedNodeSE3() {
			return solved_node_se3.Slice();
		}
		GArrayView<DualQuaternion> SolvedNodeSE3ReadOnly() const {
			return solved_node_se3.View();
		}
		GArraySlice<ushort4> PontentialPixelPair(const unsigned cam_idx) {
			return potential_pixel_pair[cam_idx].Slice();
		}
		GArrayView<ushort4> PontentialPixelPairReadOnly(const unsigned cam_idx) const {
			return potential_pixel_pair[cam_idx].View();
		}
		void Resize(const unsigned num_node) {
			solved_node_se3.ResizeArrayOrException(num_node);
		}
		void ResizePotentialPixelPair(const unsigned cam_idx, const unsigned num_pixel_pair) {
			potential_pixel_pair[cam_idx].ResizeArrayOrException(num_pixel_pair);
		}

		// Buffer
		GBufferArray<DualQuaternion> solved_node_se3;
		// #Debug
		unsigned num_cam;
		GBufferArray<ushort4> potential_pixel_pair[d_max_cam];  // Pair between measure & geo

		void log(const unsigned buffer_idx) override {
			std::string text = stringFormat(" Optimization - frame %d - buffer: %d ", frame_idx, buffer_idx);
			std::string aligned_text = stringAlign2Center(text, logging_length, "=");
			std::cout << aligned_text << std::endl;
		};
	};

}