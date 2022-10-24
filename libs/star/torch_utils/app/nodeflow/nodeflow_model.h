#pragma once
#include "torch_cpp/torch_model.h"
// NodeFlow model is PYG model, thus scatter&sparse is required
#include <torchscatter/scatter.h>
#include <torchsparse/sparse.h>

#define ENABLE_VIS_DEBUG
#ifdef ENABLE_VIS_DEBUG
#include "context.hpp"
#include "graph_visualizer.hpp"
#endif

namespace star { namespace nn {

	class NodeFlowModel : public TorchModel {
	public:
		using Ptr = std::shared_ptr<NodeFlowModel>;
		STAR_NO_COPY_ASSIGN_MOVE(NodeFlowModel);
		NodeFlowModel() { m_model_device = ""; }
		~NodeFlowModel() {};

	public:
		void Load(const std::string& model_path) override;
		void Run() override;
		torch::Tensor Output() override { return m_output; }
		// Output-related
	public:
		void SanityCheck() override;
	private:
		torch::Tensor m_output;

#ifdef ENABLE_VIS_DEBUG
	public:
		void VisualizeGT(const torch::Tensor& gt_motion);
		void VisualizeMask(const torch::Tensor& mask);
#endif
	};

}
}