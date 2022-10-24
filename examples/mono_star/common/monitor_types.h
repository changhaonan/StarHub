#pragma once
#include "frame_buffer.h"

namespace star {

	struct MonitorBuffer : public FrameBuffer {

		void create() override {
		}
		void release() override {
		}

		void log(const unsigned buffer_idx) override {
			std::string text = stringFormat(" Monitor - frame %d - buffer: %d ", frame_idx, buffer_idx);
			std::string aligned_text = stringAlign2Center(text, logging_length, "=");
			std::cout << aligned_text << std::endl;
		};
	};

}