#include "torch_cpp/app/nodeflow/nodeflow_model.h"

void star::nn::NodeFlowModel::Load(const std::string& model_path) {
	m_model = torch::jit::load(model_path);
	if (m_model_device != "")
		m_model.to(m_model_device);
	std::cout << "Model successfully loaded from " << model_path << std::endl;
}

void star::nn::NodeFlowModel::Run() {
	auto outputs = m_model.forward(m_inputs).toTuple();
	// Ouput: (Mu, Sigma)
	m_output = outputs->elements()[0].toTensor(); // at::Tensor 
}

void star::nn::NodeFlowModel::SanityCheck() {
	// Check a value
	std::cout << "234 vel: " << m_output.slice(1, 0, 3).slice(0, 234, 235) << "\n";
	std::cout << "234 pos: " << m_inputs[0].toTensor().to("cpu").slice(1, 0, 3).slice(0, 234, 235) << std::endl;
	// Check size
	//std::cout << "Output shape is:" << std::endl;
	std::cout << m_output.sizes() << std::endl;
	//std::cout << m_output.size() << "," << m_output.size(1) << "," << m_output.size(2) << std::endl;
}

#ifdef ENABLE_VIS_DEBUG
void star::nn::NodeFlowModel::VisualizeGT(const torch::Tensor& gt_motion) {
	auto& context = Easy3DViewer::Context::Instance();
	context.open(1);
	std::string graph_name = "motion_pred";
	context.addGraph(graph_name, graph_name,
		Eigen::Matrix4f::Identity(), 0.f, 1.f, 1.f, false,
		0.01f, "green");
	// Generate graph
	const size_t graph_size = m_output.sizes()[0];
	// Position
	torch::Tensor node_pos = m_inputs[0].toTensor().to("cpu");
	std::vector<float> h_node_pos(
		node_pos.data_ptr<float>(),
		node_pos.data_ptr<float>() + node_pos.numel());
	// Edge
	torch::Tensor edge_pair = m_inputs[3].toTensor().to("cpu").transpose(0, 1).to(torch::kInt32);
	std::vector<int> h_edge_pair(
		edge_pair.data_ptr<int>(),
		edge_pair.data_ptr<int>() + edge_pair.numel());

	// Motion
	torch::Tensor motion_pred = m_output.slice(1, 0, 3).to("cpu");
	std::vector<float> h_motion_pred(
		motion_pred.data_ptr<float>(),
		motion_pred.data_ptr<float>() + motion_pred.numel());
	
	Easy3DViewer::SaveGraph(
		graph_size,
		h_edge_pair.size() / 2,
		h_node_pos.data(),
		(float*)nullptr,
		h_motion_pred.data(),
		h_edge_pair.data(),
		(float*)nullptr,
		context.at(graph_name)
	);

	// GT-Motion
	std::vector<float> h_gt_motion(
		gt_motion.data_ptr<float>(),
		gt_motion.data_ptr<float>() + gt_motion.numel());

	graph_name = "gt_motion";
	context.addGraph(graph_name, graph_name,
		Eigen::Matrix4f::Identity(), 0.f, 1.f, 1.f, false,
		0.01f, "red");

	Easy3DViewer::SaveGraph(
		graph_size,
		h_edge_pair.size() / 2,
		h_node_pos.data(),
		(float*)nullptr,
		h_gt_motion.data(),
		h_edge_pair.data(),
		(float*)nullptr,
		context.at(graph_name)
	);

	// Finish
	context.close();
}

void star::nn::NodeFlowModel::VisualizeMask(const torch::Tensor& mask) {
	auto& context = Easy3DViewer::Context::Instance();
	context.open(1);
	std::string graph_name = "motion_pred";
	context.addGraph(graph_name, graph_name,
		Eigen::Matrix4f::Identity(), 0.f, 1.f, 1.f, true,
		0.01f, "green");
	// Generate graph
	const size_t graph_size = m_output.sizes()[0];
	// Position
	torch::Tensor node_pos = m_inputs[0].toTensor().to("cpu");
	std::vector<float> h_node_pos(
		node_pos.data_ptr<float>(),
		node_pos.data_ptr<float>() + node_pos.numel());
	
	// Vertex weight
	std::vector<float> invisible(
		mask.data_ptr<float>(),
		mask.data_ptr<float>() + mask.numel());
	// Edge
	torch::Tensor edge_pair = m_inputs[3].toTensor().to("cpu").transpose(0, 1).to(torch::kInt32);
	std::vector<int> h_edge_pair(
		edge_pair.data_ptr<int>(),
		edge_pair.data_ptr<int>() + edge_pair.numel());

	// Motion
	torch::Tensor motion_pred = m_output.slice(1, 0, 3).to("cpu");
	std::vector<float> h_motion_pred(
		motion_pred.data_ptr<float>(),
		motion_pred.data_ptr<float>() + motion_pred.numel());

	Easy3DViewer::SaveGraph(
		graph_size,
		h_edge_pair.size() / 2,
		h_node_pos.data(),
		invisible.data(),
		h_motion_pred.data(),
		h_edge_pair.data(),
		(float*)nullptr,
		context.at(graph_name)
	);

	// Finish
	context.close();
}
#endif