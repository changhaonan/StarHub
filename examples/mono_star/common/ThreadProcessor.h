#pragma once
#include <star/common/macro_utils.h>
#include <star/common/common_types.h>
#include "StarStageBuffer.h"

namespace star
{

	/* \brief Base Processor
	 */
	class ThreadProcessor
	{
	public:
		using Ptr = std::shared_ptr<ThreadProcessor>;
		STAR_NO_COPY_ASSIGN_MOVE(ThreadProcessor);
		ThreadProcessor(){};
		~ThreadProcessor(){};

	public:
		// Process can not change the past, only the present
		virtual void Process(
			StarStageBuffer &star_stage_buffer_this,
			const StarStageBuffer &star_stage_buffer_prev,
			cudaStream_t stream,
			const unsigned frame_idx)
		{
			printf("This is a place holder for Process.\n");
		};
		// Shared Process can change the present & past at the same time
		virtual void SharedProcess(
			StarStageBuffer &star_stage_buffer_this,
			StarStageBuffer &star_stage_buffer_prev,
			cudaStream_t stream,
			const unsigned frame_idx)
		{
			printf("This is a place holder for Shared Process.\n");
		};
		virtual void BindBuffer(
			StarStageBuffer &star_stage_buffer_0,
			StarStageBuffer &star_stage_buffer_1)
		{
			printf("This is a place holder for BindBuffer.\n");
		};
	};

}