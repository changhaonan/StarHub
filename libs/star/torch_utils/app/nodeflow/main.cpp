#include "torch_cpp/app/nodeflow/nodeflow_model.h"
#include <ctime>

using namespace star::nn;

int main(int argc, const char* argv[]) {
    //const std::string device_string = "cuda:0";
    //const std::string model_path = "C:/Users/50367/source/repos/StarNN/StarNN/data/nodeflow/traced_rstar_model_gpu.pt";
    const std::string device_string = "cpu";
    const std::string data_path = "C:/Users/50367/source/repos/Star/Star/test_data/models/test_input_values_50.pt";
    //const std::string data_path = "C:/Users/50367/source/repos/Star/Star/test_data/models/test_input_values_100.pt";
    //const std::string data_path = "C:/Users/50367/source/repos/Star/Star/test_data/models/test_input_values_300.pt";
    //const std::string data_path = "C:/Users/50367/source/repos/Star/Star/test_data/models/test_input_values_500.pt";
    //const std::string data_path = "C:/Users/50367/source/repos/Star/Star/test_data/models/test_input_values_900.pt";

//    const std::string model_path = "C:/Users/50367/source/repos/Star/Star/test_data/models/traced_occlusionfu_model_cpu.pt";
//#ifdef ENABLE_VIS_DEBUG
//    auto& context = Easy3DViewer::Context::Instance();
//    context.setDir("C:/Users/50367/source/repos/Star/Star/external/Easy3DViewer/public/test_data/occlusionfusion", "frame");
//    context.clearDir();
//#endif

    const std::string model_path = "C:/Users/50367/source/repos/Star/Star/test_data/models/traced_rstar_50epc_8bs_model_cpu.pt";
#ifdef ENABLE_VIS_DEBUG
    auto& context = Easy3DViewer::Context::Instance();
    context.setDir("C:/Users/50367/source/repos/Star/Star/external/Easy3DViewer/public/test_data/r_star", "frame");
    context.clearDir();
#endif

    NodeFlowModel::Ptr nodeflow_model = std::make_shared<NodeFlowModel>();
    nodeflow_model->SetDevice(device_string);
    nodeflow_model->Load(model_path);
 
    // prepare input data
    torch::jit::script::Module container = torch::jit::load(data_path);
    torch::Tensor node_pos = container.attr("node_pos").toTensor().to(device_string);
    torch::Tensor curr_motion = container.attr("curr_motion").toTensor().to(device_string);
    torch::Tensor historical_motion = container.attr("historical_motion").toTensor().to(device_string);
    torch::Tensor edge_indices_0 = container.attr("edge_indices_0").toTensor().to(device_string);
    torch::Tensor edge_indices_1 = container.attr("edge_indices_1").toTensor().to(device_string);
    torch::Tensor edge_indices_2 = container.attr("edge_indices_2").toTensor().to(device_string);
    torch::Tensor edge_indices_3 = container.attr("edge_indices_3").toTensor().to(device_string);
    torch::Tensor down_sample_indices_0 = container.attr("down_sample_indices_0").toTensor().to(device_string);
    torch::Tensor down_sample_indices_1 = container.attr("down_sample_indices_1").toTensor().to(device_string);
    torch::Tensor down_sample_indices_2 = container.attr("down_sample_indices_2").toTensor().to(device_string);
    torch::Tensor up_sample_indices_0 = container.attr("up_sample_indices_0").toTensor().to(device_string);
    torch::Tensor up_sample_indices_1 = container.attr("up_sample_indices_1").toTensor().to(device_string);
    torch::Tensor up_sample_indices_2 = container.attr("up_sample_indices_2").toTensor().to(device_string);
    torch::Tensor invisible_mask = container.attr("invisible_mask").toTensor().to(device_string);
    //torch::Tensor gt_motion = container.attr("gt_motion").toTensor().to(device_string);
    torch::Tensor h0 = container.attr("h0").toTensor().to(device_string);
    torch::Tensor c0 = container.attr("c0").toTensor().to(device_string);

    auto& inputs = nodeflow_model->Inputs();
    inputs.push_back(node_pos);
    inputs.push_back(curr_motion);
    inputs.push_back(historical_motion);
    inputs.push_back(edge_indices_0);
    inputs.push_back(edge_indices_1);
    inputs.push_back(edge_indices_2);
    inputs.push_back(edge_indices_3);
    inputs.push_back(down_sample_indices_0);
    inputs.push_back(down_sample_indices_1);
    inputs.push_back(down_sample_indices_2);
    inputs.push_back(up_sample_indices_0);
    inputs.push_back(up_sample_indices_1);
    inputs.push_back(up_sample_indices_2);
    inputs.push_back(h0);
    inputs.push_back(c0);
    //inputs.push_back(gt_motion);
    
    // Run
    nodeflow_model->Run();
    nodeflow_model->SanityCheck();

#ifdef ENABLE_VIS_DEBUG
    nodeflow_model->VisualizeMask(invisible_mask);
#endif

    return 0;
}