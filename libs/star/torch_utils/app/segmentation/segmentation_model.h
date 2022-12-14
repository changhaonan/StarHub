#pragma once
#include <star/torch_utils/torch_model.h>
#include <torch/script.h> // One-stop header.
#include <torch/torch.h>

#include <type_traits>
#include <iostream>
#include <memory>

#define ENABLE_VIS_DEBUG
#ifdef ENABLE_VIS_DEBUG
#include <easy3d_viewer/context.hpp>
#include <easy3d_viewer/graph_visualizer.hpp>
#endif

namespace star::nn
{

	class SegmentationModel : public TorchModel
	{
	public:
		using Ptr = std::shared_lock<SegmentationModel>;
		STAR_NO_COPY_ASSIGN_MOVE(SegmentationModel);
		SegmentationModel() { m_model_device = ""; }
		~SegmentationModel(){};

		void Load(const std::string &model_path) override;
		void Run() override;
		torch::Tensor Output() override;
		void SanityCheck();

#ifdef ENABLE_VIS_DEBUG
		void VisualizeMask(const torch::Tensor &mask);
#endif

	private:
		torch::Tensor m_output;
	};

}