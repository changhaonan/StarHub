#include <star/torch_utils/app/opticalflow/opticalflow_model.h>
#include <ctime>

using namespace star::nn;

int main(int argc, const char *argv[])
{
    const std::string device_string = "cuda:0";
    const std::string model_path = "C:/Users/50367/source/repos/StarNN/StarNN/data/opticalflow/traced_gma_model_gpu.pt";
    // const std::string device_string = "cpu";
    // const std::string model_path = "C:/Users/50367/source/repos/StarNN/StarNN/data/opticalflow/traced_gma_model_cpu.pt";
    // const std::string image1_path = "C:/Users/50367/source/repos/StarNN/StarNN/data/opticalflow/image_1.pt";
    // const std::string image2_path = "C:/Users/50367/source/repos/StarNN/StarNN/data/opticalflow/image_2.pt";

#ifdef ENABLE_VIS_DEBUG
    auto &context = easy3d::Context::Instance();
    context.setDir("C:/Users/50367/source/repos/StarNN/Easy3DViewer/public/test_data/opticalflow", "frame");
    context.clearDir();
#endif

    TorchModel::Ptr opticalflow_model = std::make_shared<OpticalFlowModel>();
    opticalflow_model->SetDevice(device_string);
    opticalflow_model->Load(model_path);

    // prepare input data
    // torch::jit::script::Module image_1 = torch::jit::load(image1_path).toTensor().to(device_string);
    // torch::Tensor image_1 = torch::ones({ 1, 4, 218, 224 }).to(device_string);
    // torch::Tensor image_2 = torch::ones({ 1, 4, 218, 224 }).to(device_string);
    torch::jit::script::Module container = torch::jit::load("C:/Users/50367/source/repos/StarNN/StarNN/data/opticalflow/images.pt");
    torch::Tensor image_1 = container.attr("images1").toTensor().to(device_string);
    torch::Tensor image_2 = container.attr("images2").toTensor().to(device_string);

    for (auto i = 0; i < 10; ++i)
    {
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(image_1);
        inputs.push_back(image_2);
        opticalflow_model->SetInputs(inputs);
        // Run multiple times
        opticalflow_model->Run();
        opticalflow_model->SanityCheck();
    }

#ifdef ENABLE_VIS_DEBUG
    // opticalflow_model->Visualization(gt_motion);
#endif

    return 0;
}