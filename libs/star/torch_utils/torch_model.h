#pragma once
#include <torch/script.h>
#include <star/common/macro_utils.h>
#include <star/common/common_types.h>

namespace star::nn
{

	/* \brief BaseModel class
	 */
	class TorchModel
	{
	public:
		using Ptr = std::shared_ptr<TorchModel>;
		STAR_NO_COPY_ASSIGN_MOVE(TorchModel);
		TorchModel(){};
		~TorchModel(){};

	public:
		// Load model using torchscript
		virtual void Load(const std::string &model_path) = 0;
		virtual void Run() = 0;
		virtual torch::Tensor Output() = 0;
		virtual void SanityCheck() = 0;

	public:
		void SetDevice(const std::string &device_string) { m_model_device = device_string; }
		std::string GetDevice() const { return m_model_device; }
		std::vector<torch::jit::IValue> &Inputs() { return m_inputs; }
		void SetInputs(std::vector<torch::jit::IValue> &inputs) { m_inputs = inputs; }

	protected:
		torch::jit::script::Module m_model;
		std::vector<torch::jit::IValue> m_inputs;
		std::string m_model_device; // Where the model & data are loaded
	};
}