#include <star/torch_utils/app/opticalflow/opticalflow_model.h>

void star::nn::OpticalFlowModel::Load(const std::string &model_path)
{
	m_model = torch::jit::load(model_path);
	if (m_model_device != "")
		m_model.to(m_model_device);
	std::cout << "Model successfully loaded from " << model_path << std::endl;
}

void star::nn::OpticalFlowModel::Run()
{
	// Transformations
	if (m_use_undo_pad)
	{
		m_inputs[0] = doPadding(m_inputs[0]);
		m_inputs[1] = doPadding(m_inputs[1]);
		m_output = undoPadding(m_model.forward(m_inputs).toTuple()->elements()[1].toTensor()).squeeze(0).permute({1, 2, 0}).contiguous();
	}
	else
	{
		m_output = m_model.forward(m_inputs).toTuple()->elements()[1].toTensor().squeeze(0).permute({1, 2, 0}).contiguous();
	}
}

torch::jit::IValue star::nn::OpticalFlowModel::doPadding(torch::jit::IValue image)
{
	// Round image off to nearest multiple of 8 in terms of size
	c10::ArrayRef<int64_t> sizes = image.toTensor().sizes().slice(2);
	int height = sizes[0];
	int width = sizes[1];

	int pad_ht = (((height / 8) + 1) * 8 - height) % 8;
	int pad_wd = (((width / 8) + 1) * 8 - width) % 8;

	// TODO: check for mode sintel or otherwise
	m_padding = {pad_wd / 2, pad_wd - pad_wd / 2, pad_ht / 2, pad_ht - pad_ht / 2};

	return (torch::jit::IValue)torch::nn::functional::pad(image.toTensor(), torch::nn::functional::PadFuncOptions(m_padding).mode(torch::kReplicate));
};

at::Tensor star::nn::OpticalFlowModel::undoPadding(at::Tensor image)
{
	c10::ArrayRef<int64_t> sizes = image.sizes().slice(2);
	int height = sizes[0];
	int width = sizes[1];
	std::vector<int64_t> unpad = {m_padding[2], height - m_padding[3], m_padding[0], width - m_padding[1]};
	return image.index({"...", torch::indexing::Slice(unpad[0], unpad[1]), torch::indexing::Slice(unpad[2], unpad[3])});
};

void star::nn::OpticalFlowModel::SanityCheck()
{
	// Check a value
	std::cout << m_output.slice(0, 0, 1).slice(1, 0, 1) << "\n";
	// Check size
	std::cout << "Output shape is:" << std::endl;
	std::cout << m_output.sizes() << std::endl;
}

#ifdef ENABLE_VIS_DEBUG
void star::nn::OpticalFlowModel::Visualization(const torch::Tensor &gt_motion)
{
	auto &context = easy3d::Context::Instance();
	context.open(1);

	// Finish
	context.close();
}
#endif