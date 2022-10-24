#pragma once
#include <star/common/StageBuffer.h>
#include "measure_types.h"
#include "opticalflow_types.h"
#include "nodeflow_types.h"
#include "dynamic_geometry_types.h"
#include "optimization_types.h"
#include "segmentation_types.h"
#include "monitor_types.h"

namespace star
{

	/*
	 * \brief 2 stages Buffer, t, t+1 is where the index coordinate is defined
	 * Each big stage only based on previous stage & output of itself in last frame
	 * Stage-0:     Measure;          		MeasureBuffer_{t+1}, (vertex, normal, color, index)
	 * Stage-0.i:   Segmentation		 	SegmentationBuffer_t (Activate per 2 frame)
	 * Stage-I.0:   OpticalFlow;      		OpticalFlowBuffer_t, (of)
	 * Stage-I.i:   NodeFlow;		 		NodeFlowBuffer_t, (knn_map, nodeflow)
	 * Stage-I.ii:  DynamicGeometry;  		WarpField&Geometry jointly update
	 * Stage-II:	Monitor
	 */
	constexpr unsigned Measure = 0;
	constexpr unsigned Segmentation = 1;
	constexpr unsigned Process = 2;
	constexpr unsigned Monitor = 3;
	constexpr unsigned NumStage = 4;

	constexpr unsigned OpticalFlow = 0;
	constexpr unsigned NodeFlow = 1;
	constexpr unsigned Optimization = 2;
	constexpr unsigned DynamicGeometry = 3;
	constexpr unsigned NumProcessStage = 4;

	constexpr unsigned NumAllStage = NumStage + NumProcessStage - 1;

	class StarStageBuffer
	{
	public:
		StarStageBuffer()
		{
			// Measure
			m_stage_buffer.SetBuffer((void *)(new MeasureBuffer()), Measure);
			// Segmentation
			m_stage_buffer.SetBuffer((void *)(new SegmentationBuffer()), Segmentation);
			// Process
			m_stage_buffer.SetBuffer(nullptr, Process);
			m_process_stage_buffer.SetBuffer((void *)(new OpticalFlowBuffer()), OpticalFlow);
			m_process_stage_buffer.SetBuffer((void *)(new NodeFlowBuffer()), NodeFlow);
			m_process_stage_buffer.SetBuffer((void *)(new OptimizationBuffer()), Optimization);
			m_process_stage_buffer.SetBuffer((void *)(new DynamicGeometryBuffer()), DynamicGeometry);
			// Monitor
			m_stage_buffer.SetBuffer((void *)(new MonitorBuffer()), Monitor);
		};
		~StarStageBuffer()
		{
			// Measure
			m_stage_buffer.DeleteBuffer(Measure);
			// Segmentation
			m_stage_buffer.DeleteBuffer(Segmentation);
			// Process
			m_process_stage_buffer.DeleteBuffer(OpticalFlow);
			m_process_stage_buffer.DeleteBuffer(NodeFlow);
			m_process_stage_buffer.DeleteBuffer(Optimization);
			m_process_stage_buffer.DeleteBuffer(DynamicGeometry);
			// Monitor
			m_stage_buffer.DeleteBuffer(Monitor);
		}
		// Measure-related
		void MeasureWait() { m_stage_buffer.Wait(Measure); }
		void MeasureMoveOn() { m_stage_buffer.MoveOn(); }
		MeasureBuffer *GetMeasureBuffer() { return m_stage_buffer.Buffer<MeasureBuffer>(Measure); } // Order is important
		const MeasureBuffer *GetMeasureBufferReadOnly() const { return m_stage_buffer.BufferReadOnly<MeasureBuffer>(Measure); }

		// Pre-process-related
		void SegmentationWait() { m_stage_buffer.Wait(Segmentation); }
		void SegmentationMoveOn() { m_stage_buffer.MoveOn(); }
		SegmentationBuffer *GetSegmentationBuffer() { return m_stage_buffer.Buffer<SegmentationBuffer>(Segmentation); } // Order is important
		const SegmentationBuffer *GetSegmentationBufferReadOnly() const { return m_stage_buffer.BufferReadOnly<SegmentationBuffer>(Segmentation); }

		// OpticalFlow-related
		void OpticalFlowWait()
		{
			m_process_stage_buffer.Wait(OpticalFlow);
			m_stage_buffer.Wait(Process);
		}
		void OpticalFlowMoveOn() { m_process_stage_buffer.MoveOn(); }
		OpticalFlowBuffer *GetOpticalFlowBuffer() { return m_process_stage_buffer.Buffer<OpticalFlowBuffer>(OpticalFlow); }
		const OpticalFlowBuffer *GetOpticalFlowBufferReadOnly() const { return m_process_stage_buffer.BufferReadOnly<OpticalFlowBuffer>(OpticalFlow); }

		// NodeFlow-realted
		void NodeFlowWait() { m_process_stage_buffer.Wait(NodeFlow); }
		void NodeFlowMoveOn() { m_process_stage_buffer.MoveOn(); }
		NodeFlowBuffer *GetNodeFlowBuffer() { return m_process_stage_buffer.Buffer<NodeFlowBuffer>(NodeFlow); }
		const NodeFlowBuffer *GetNodeFlowBufferReadOnly() const { return m_process_stage_buffer.BufferReadOnly<NodeFlowBuffer>(NodeFlow); }

		// Optimization-realted
		void OptimizationWait() { m_process_stage_buffer.Wait(Optimization); }
		void OptimizationMoveOn() { m_process_stage_buffer.MoveOn(); }
		OptimizationBuffer *GetOptimizationBuffer() { return m_process_stage_buffer.Buffer<OptimizationBuffer>(Optimization); }
		const OptimizationBuffer *GetOptimizationBufferReadOnly() const { return m_process_stage_buffer.BufferReadOnly<OptimizationBuffer>(Optimization); }

		// Optimization-realted
		void DynamicGeometryWait() { m_process_stage_buffer.Wait(DynamicGeometry); }
		void DynamicGeometryMoveOn()
		{
			m_stage_buffer.MoveOn();
			m_process_stage_buffer.MoveOn();
		} // Order is important
		DynamicGeometryBuffer *GetDynamicGeometryBuffer() { return m_process_stage_buffer.Buffer<DynamicGeometryBuffer>(DynamicGeometry); }
		const DynamicGeometryBuffer *GetDynamicGeometryBufferReadOnly() const { return m_process_stage_buffer.BufferReadOnly<DynamicGeometryBuffer>(DynamicGeometry); }

		// Monitor-related, no buffer bounded
		void MonitorWait() { m_stage_buffer.Wait(Monitor); }
		void MonitorMoveOn() { m_stage_buffer.MoveOn(); }
		MonitorBuffer *GetMonitorBuffer() { return m_stage_buffer.Buffer<MonitorBuffer>(Monitor); }
		const MonitorBuffer *GetMonitorBufferReadOnly() const { return m_stage_buffer.BufferReadOnly<MonitorBuffer>(Monitor); }

	private:
		// Major stage: Measure, PreProcess, Process, Monitor
		StageBuffer<NumStage> m_stage_buffer;
		// Sub stages inside Process stage
		StageBuffer<NumProcessStage> m_process_stage_buffer;
	};

}