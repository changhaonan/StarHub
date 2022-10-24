#pragma once
#include <chrono>
#include <star/common/string_utils.hpp>

namespace star
{

	struct FrameBuffer
	{
		enum
		{
			logging_length = 50
		};
		FrameBuffer()
		{
			frame_idx = 0;
			create();
		};
		~FrameBuffer() { release(); }
		unsigned frame_idx;
		unsigned frame() const { return frame_idx; }
		void step(const unsigned frame_step) { frame_idx += frame_step; }
		virtual void create(){};
		virtual void release(){};
		virtual void log(const unsigned buffer_idx)
		{
			std::string text = stringFormat(" Buffer - frame %d - buffer: %d ", frame_idx, buffer_idx);
			std::string aligned_text = stringAlign2Center(text, logging_length, "=");
			std::cout << aligned_text << std::endl;
		};
		// Timer
		void tic()
		{
			time_mark = std::chrono::high_resolution_clock::now();
		}
		void toc()
		{
			auto now = std::chrono::high_resolution_clock::now();
			auto time = now - time_mark;
			std::cout << "tik-toc " << time / std::chrono::milliseconds(1) << " ms" << std::endl;
		}
		std::chrono::time_point<std::chrono::high_resolution_clock> time_mark;
	};
}