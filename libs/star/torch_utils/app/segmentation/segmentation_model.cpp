#include <star/torch_utils/app/segmentation/segmentation_model.h>

void star::nn::SegmentationModel::Load(const std::string &model_path)
{
	m_model = torch::jit::load(model_path);
	if (m_model_device != "")
		m_model.to(m_model_device);
	std::cout << "Model successfully loaded from " << model_path << std::endl;
}

void star::nn::SegmentationModel::Run()
{
	try
	{
		m_output = m_model.forward(m_inputs).toTensor().permute({0, 2, 3, 1}).contiguous();
	}
	catch (std::exception &e)
	{
		std::cerr << e.what() << std::endl;
	}
}

torch::Tensor star::nn::SegmentationModel::Output()
{
	// Get the maximal likelihood result
	try
	{
		auto result = torch::nn::functional::softmax(m_output, 3);
		result = result.argmax(3).to(torch::kInt32); // torch.long to torch.int
		return result;
	}
	catch (std::exception &e)
	{
		std::cout << e.what() << std::endl;
	}
	return torch::Tensor();
}

void star::nn::SegmentationModel::VisualizeMask(const torch::Tensor &mask)
{
}

void star::nn::SegmentationModel::SanityCheck()
{
	// std::cout << "Segmenter sanity checking..." << std::endl;
	// std::cout << m_output << std::endl;
}