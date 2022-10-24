#pragma once
#include "solver_types.h"
#include "frame_buffer.h"

namespace star {

	struct SingleOpticalFlowMap {
		cudaTextureObject_t opticalflow_map;
	};


	struct SingleOpticalFlowBuffer {
		CudaTextureSurface opticalflow;
		void create(const unsigned img_rows, const unsigned img_cols) {
			createFloat2TextureSurface(img_rows, img_cols, opticalflow);
		}
		void release() {
			releaseTextureCollect(opticalflow);
		}
		operator SingleOpticalFlowMap () const {  // Auto-transfer
			SingleOpticalFlowMap  opticalflow_map;
			opticalflow_map.opticalflow_map = opticalflow.texture;
			return opticalflow_map;
		}
	};


	struct OpticalFlowBuffer : FrameBuffer {
		OpticalFlowBuffer() { create(); }
		~OpticalFlowBuffer() { release(); }
		void create() override {
			auto& config = ConfigParser::Instance();
			for (auto cam_idx = 0; cam_idx < config.num_cam(); ++cam_idx) {
				buffer[cam_idx].create(
					config.downsample_img_rows(cam_idx),
					config.downsample_img_cols(cam_idx));
			}
		}
		void release() override {
			auto& config = ConfigParser::Instance();
			for (auto cam_idx = 0; cam_idx < config.num_cam(); ++cam_idx) {
				buffer[cam_idx].release();
			}
		}
		
		SingleOpticalFlowBuffer buffer[d_max_cam];

		void log(const unsigned buffer_idx) override {
			std::string text = stringFormat(" OpticalFlow - frame %d - buffer: %d ", frame_idx, buffer_idx);
			std::string aligned_text = stringAlign2Center(text, logging_length, "=");
			std::cout << aligned_text << std::endl;
		};

		// Solver-API
		OpticalFlow4Solver GenerateOpticalFlow4Solver(const unsigned num_cam) const {
			OpticalFlow4Solver opticalflow4solver;
			for (auto cam_idx = 0; cam_idx < num_cam; ++cam_idx) {
				opticalflow4solver.opticalflow_map[cam_idx] =
					buffer[cam_idx].opticalflow.texture;
			}
			opticalflow4solver.num_cam = num_cam;
			return opticalflow4solver;
		}
	};

}