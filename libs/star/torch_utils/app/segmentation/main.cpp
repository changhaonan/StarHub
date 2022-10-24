#include "torch_cpp/app/nodeflow/nodeflow_model.h"
#include "torch_cpp/app/segmentation/segmentation_model.h"
#include <ctime>

using namespace star::nn;

int main(int argc, const char* argv[]) {
	//const std::string device_string = "cuda:0";
	//const std::string model_path = "C:/Users/50367/source/repos/StarCore/Star/test_data/models/segmenter_320x240_model_gpu.pt";
	
	const std::string device_string = "cpu";
	const std::string model_path = "C:/Users/50367/source/repos/StarCore/Star/test_data/models/segmenter_320x240_model_cpu.pt";

	TorchModel::Ptr segmentation_model = std::make_shared<SegmentationModel>();
	segmentation_model->SetDevice(device_string);
	segmentation_model->Load(model_path);

	torch::jit::script::Module container = torch::jit::load("C:/Users/50367/source/repos/StarCore/Star/test_data/models/segmenter_320x240_test_image.pt");
	torch::Tensor image = container.attr("image").toTensor().to(device_string).unsqueeze(0);
	std::cout << image.sizes() << std::endl;
	for (auto i = 0; i < 10; ++i) {
		std::vector<torch::jit::IValue> inputs;
		inputs.push_back(image);
		segmentation_model->SetInputs(inputs);
		// Run multiple times
		segmentation_model->Run();
		segmentation_model->SanityCheck();
	}

	return 0;
}