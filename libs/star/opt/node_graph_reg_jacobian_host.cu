#include <star/opt/node_graph_reg_jacobian_host.h>

/** Diagonal JtJ
 */
void star::updateRegJtJDiagonalHost(
	std::vector<float> &jtj_flatten,
	const NodeGraphRegTerm2Jacobian &node_graph_reg_term2jacobian,
	const float weight_square,
	const bool verbose)
{
	// Download the required data
	std::vector<float3> Ti_xj;
	std::vector<float3> Tj_xj;
	std::vector<ushort3> node_graph;
	std::vector<float> connect_weight;
	std::vector<unsigned char> validity;
	node_graph_reg_term2jacobian.Ti_xj.Download(Ti_xj);
	node_graph_reg_term2jacobian.Tj_xj.Download(Tj_xj);
	node_graph_reg_term2jacobian.node_graph.Download(node_graph);
	node_graph_reg_term2jacobian.connect_weight.Download(connect_weight);
	node_graph_reg_term2jacobian.validity_indicator.Download(validity);

	// Iterates through all node pairs
	for (auto i = 0; i < node_graph.size(); i++)
	{
		if (validity[i] == 0)
			continue;
		ushort3 node_ij_k = node_graph[i];
		float weight = connect_weight[i];
		GradientOfScalarCost gradient_i[3];
		GradientOfScalarCost gradient_j[3];
		float residual[3];
		device::computeRegTermJacobianResidual(Ti_xj[i], Tj_xj[i], weight, residual, gradient_i, gradient_j);

		if (node_ij_k.x == node_ij_k.y)
			continue;
		// First fill the node i
		float *jtj = &jtj_flatten[size_t(node_ij_k.x) * d_node_variable_dim_square];
		for (auto channel = 0; channel < 3; channel++)
		{
			float *jacobian = (float *)(&gradient_i[channel]);
			for (int jac_row = 0; jac_row < d_node_variable_dim; jac_row++)
			{
				for (int jac_col = 0; jac_col < d_node_variable_dim; jac_col++)
				{
					jtj[d_node_variable_dim * jac_row + jac_col] += weight_square * jacobian[jac_col] * jacobian[jac_row];

					//[Debug]
					if (verbose)
					{
						printf("CPU: term: %d, val: %f, reg jacobian at node_%d: (%d, %d) is : (%f, %f, %f, %f, %f, %f).\n",
							   i,
							   jtj[d_node_variable_dim * jac_row + jac_col],
							   node_ij_k.x, jac_row, jac_col,
							   jacobian[0], jacobian[1], jacobian[2],
							   jacobian[3], jacobian[4], jacobian[5]);
					}
				}
			}
		}

		// Then fill node j
		jtj = &jtj_flatten[size_t(node_ij_k.y) * d_node_variable_dim_square];
		for (auto channel = 0; channel < 3; channel++)
		{
			float *jacobian = (float *)(&gradient_j[channel]);
			for (int jac_row = 0; jac_row < d_node_variable_dim; jac_row++)
			{
				for (int jac_col = 0; jac_col < d_node_variable_dim; jac_col++)
				{
					jtj[d_node_variable_dim * jac_row + jac_col] += weight_square * jacobian[jac_col] * jacobian[jac_row];

					////[Debug]
					// if ((node_ij_k.y * d_node_variable_dim_square + d_node_variable_dim * jac_row + jac_col) == 0) {
					//	printf("CPU: term: %d, val: %f, reg jacobian at node_%d: (%d, %d) is : (%f, %f, %f, %f, %f, %f).\n",
					//		i,
					//		jtj[d_node_variable_dim * jac_row + jac_col],
					//		node_ij_k.y, jac_row, jac_col,
					//		jacobian[0], jacobian[1], jacobian[2],
					//		jacobian[3], jacobian[4], jacobian[5]);
					// }
				}
			}
		}
	}
}

/** Non-Diagonal JtJ
 */
void star::updateRegJtJBlockHost(
	const GArrayView<unsigned> &encoded_nodepair,
	std::vector<float> &jtj_flatten,
	const NodeGraphRegTerm2Jacobian &node_graph_reg_term2jacobian,
	const float weight_square)
{

	// Download the required data
	std::vector<float3> Ti_xj;
	std::vector<float3> Tj_xj;
	std::vector<ushort3> node_graph;
	std::vector<float> connect_weight;
	std::vector<unsigned char> validity_indicator;
	node_graph_reg_term2jacobian.Ti_xj.Download(Ti_xj);
	node_graph_reg_term2jacobian.Tj_xj.Download(Tj_xj);
	node_graph_reg_term2jacobian.node_graph.Download(node_graph);
	node_graph_reg_term2jacobian.connect_weight.Download(connect_weight);
	node_graph_reg_term2jacobian.validity_indicator.Download(validity_indicator);

	// Prepare the map from node pair to index
	std::map<unsigned, unsigned> pair2index;
	pair2index.clear();
	std::vector<unsigned> encoded_pair_vec;
	encoded_nodepair.Download(encoded_pair_vec);

	for (auto i = 0; i < encoded_pair_vec.size(); i++)
	{
		auto encoded_pair = encoded_pair_vec[i];
		// There should not be duplicate
		auto iter = pair2index.find(encoded_pair);
		assert(iter == pair2index.end());

		// insert it
		pair2index.insert(std::make_pair(encoded_pair, i));
	}

	for (auto term_idx = 0; term_idx < node_graph.size(); term_idx++)
	{
		unsigned char validity = validity_indicator[term_idx];
		if (!validity)
			continue; // Jump the invalid

		ushort3 node_ij_k = node_graph[term_idx];
		float connect_weight_sqaure = connect_weight[term_idx] * connect_weight[term_idx];
		GradientOfScalarCost gradient_i[3];
		GradientOfScalarCost gradient_j[3];
		device::computeRegTermJacobian(Ti_xj[term_idx], Tj_xj[term_idx], gradient_i, gradient_j);

		// First fill the (node i, node j)
		const unsigned node_i = node_ij_k.x;
		const unsigned node_j = node_ij_k.y;
		if (node_i == node_j)
			continue;
		const float *encoded_jacobian_i = (const float *)(gradient_i);
		const float *encoded_jacobian_j = (const float *)(gradient_j);
		auto encoded_ij = encode_nodepair(node_i, node_j);
		auto iter = pair2index.find(encoded_ij);
		if (iter != pair2index.end())
		{
			unsigned index = iter->second;
			float *jtj = &jtj_flatten[size_t(index) * d_node_variable_dim_square];
			for (auto channel = 0; channel < 3; channel++)
			{
				const float *jacobian_i = encoded_jacobian_i + size_t(channel) * d_node_variable_dim;
				const float *jacobian_j = encoded_jacobian_j + size_t(channel) * d_node_variable_dim;
				for (int jac_row = 0; jac_row < d_node_variable_dim; jac_row++)
				{
					for (int jac_col = 0; jac_col < d_node_variable_dim; jac_col++)
					{
						jtj[d_node_variable_dim * jac_row + jac_col] += connect_weight_sqaure * weight_square * jacobian_i[jac_col] * jacobian_j[jac_row];
					}
				}
			}
		}
		else
		{
			// Kill it
			LOG(FATAL) << "Cannot find the index for node " << node_i << " and " << node_j << " pair in term: " << term_idx << ".";
		}

		// Next fill (node j, node i)
		encoded_jacobian_i = (const float *)(gradient_j);
		encoded_jacobian_j = (const float *)(gradient_i);
		encoded_ij = encode_nodepair(node_j, node_i);
		iter = pair2index.find(encoded_ij);
		if (iter != pair2index.end())
		{
			unsigned index = iter->second;
			float *jtj = &jtj_flatten[size_t(index) * d_node_variable_dim_square];
			for (auto channel = 0; channel < 3; channel++)
			{
				const float *jacobian_i = encoded_jacobian_i + size_t(channel) * d_node_variable_dim;
				const float *jacobian_j = encoded_jacobian_j + size_t(channel) * d_node_variable_dim;
				for (int jac_row = 0; jac_row < d_node_variable_dim; jac_row++)
				{
					for (int jac_col = 0; jac_col < d_node_variable_dim; jac_col++)
					{
						jtj[d_node_variable_dim * jac_row + jac_col] += connect_weight_sqaure * weight_square * jacobian_i[jac_col] * jacobian_j[jac_row];
					}
				}
			}
		}
		else
		{
			// Kill it
			LOG(FATAL) << "Cannot find the index for node " << node_i << " and " << node_j << " pair in term: " << term_idx << ".";
		}
	}
}

/** JtResidual
 */
void star::updateRegJtResidualHost(
	std::vector<float> &jt_residual,
	const NodeGraphRegTerm2Jacobian &node_graph_reg_term2jacobian,
	const float weight_square)
{
	// Download the required data
	std::vector<float3> Ti_xj;
	std::vector<float3> Tj_xj;
	std::vector<ushort3> node_graph;
	std::vector<float> connect_weight;
	std::vector<unsigned char> validity;
	node_graph_reg_term2jacobian.Ti_xj.Download(Ti_xj);
	node_graph_reg_term2jacobian.Tj_xj.Download(Tj_xj);
	node_graph_reg_term2jacobian.node_graph.Download(node_graph);
	node_graph_reg_term2jacobian.connect_weight.Download(connect_weight);
	node_graph_reg_term2jacobian.validity_indicator.Download(validity);

	// Iterates through all node pairs
	for (auto k = 0; k < node_graph.size(); k++)
	{
		if (validity[k] == 0)
			continue; // Jump invalid pair

		ushort3 node_ij_k = node_graph[k];
		float jt_residual_array[d_node_variable_dim] = {0.f};
		// First fill the node i
		device::computeRegTermJtResidual(
			Ti_xj[k], Tj_xj[k],
			connect_weight[k],
			true,
			jt_residual_array);
		float *jt_r_node = &jt_residual[size_t(node_ij_k.x) * d_node_variable_dim];
		for (auto j = 0; j < d_node_variable_dim; j++)
		{
			jt_r_node[j] += -weight_square * jt_residual_array[j];
		}

		// First fill the node j
		device::computeRegTermJtResidual(
			Ti_xj[k], Tj_xj[k],
			connect_weight[k],
			false,
			jt_residual_array);
		jt_r_node = &jt_residual[size_t(node_ij_k.y) * d_node_variable_dim];
		for (auto j = 0; j < d_node_variable_dim; j++)
		{
			jt_r_node[j] += -weight_square * jt_residual_array[j];
		}
	}
}