#pragma once
#include "frame_buffer.h"
#include "fusion_types.h"

namespace star {

	struct SegmentationBuffer : public FrameBuffer {
		SegmentationBuffer() { create(); }
		~SegmentationBuffer() { release(); }

		void create() override {
			auto& config = ConfigParser::Instance();
			num_cam = config.num_cam();
			for (auto cam_idx = 0; cam_idx < num_cam; ++cam_idx) {
				auto image_width = config.downsample_img_cols(cam_idx);
				auto image_height = config.downsample_img_rows(cam_idx);
				//auto image_width = config.raw_img_cols(cam_idx);
				//auto image_height = config.raw_img_rows(cam_idx);
				createInt32TextureSurface(image_height, image_width, segmentation[cam_idx]);	// Uint32
			}
		}
		void release() override {
			for (auto cam_idx = 0; cam_idx < num_cam; ++cam_idx) {
				releaseTextureCollect(segmentation[cam_idx]);
			}
		}

		void log(const unsigned buffer_idx) override {
			std::string text = stringFormat(" Segmentation - frame %d - buffer: %d ", frame_idx, buffer_idx);
			std::string aligned_text = stringAlign2Center(text, logging_length, "=");
			std::cout << aligned_text << std::endl;
		};

		Segmentation4SemanticFusion GenerateSegmentation4SemanticFusion() const {
			Segmentation4SemanticFusion segmentation4semantic_fusion{};
			for (auto cam_idx = 0; cam_idx < num_cam; ++cam_idx) {
				segmentation4semantic_fusion.segmentation[cam_idx] = segmentation[cam_idx].texture;
			}
			return segmentation4semantic_fusion;
		}

		unsigned num_cam;
		// Buffer
		CudaTextureSurface segmentation[d_max_cam];
	};
}