#pragma once
#include "torch_cpp/torch_model.h"
#include <torch/script.h> // One-stop header.
#include <torch/torch.h>

#include <type_traits>
#include <iostream>
#include <memory>

#define ENABLE_VIS_DEBUG
#ifdef ENABLE_VIS_DEBUG
#include "context.hpp"
#include "graph_visualizer.hpp"
#endif

namespace star { namespace nn {

	class OpticalFlowModel : public TorchModel {
	public:
		using Ptr = std::shared_ptr<OpticalFlowModel>;
		STAR_NO_COPY_ASSIGN_MOVE(OpticalFlowModel);
		OpticalFlowModel() : m_use_undo_pad(false) { m_model_device = ""; }
		~OpticalFlowModel() {};

		void Load(const std::string& model_path) override;
		void Run() override;
		torch::Tensor Output() override { return m_output; }
		// Check
		void SanityCheck() override;

#ifdef ENABLE_VIS_DEBUG
		void Visualization(const torch::Tensor& gt_motion);
#endif

	private:
		// Padding operations
		torch::jit::IValue doPadding(torch::jit::IValue image);
		at::Tensor undoPadding(torch::Tensor image);

		torch::Tensor m_output;
		std::vector<int64_t> m_padding;
		bool m_use_undo_pad;
	};

}
}