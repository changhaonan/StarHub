#include <star/opt/dense_image_jacobian_host.h>

/** Diagonal Jacobian in Debug
 */
void star::updateDenseImageJtJDiagonalHost(
	std::vector<float> &jtj_flatten,
	DenseImageTerm2Jacobian &term2jacobian,
	const floatX<d_dense_image_residual_dim> &weight_square_vec,
	const unsigned inspect_index,
	const bool verbose)
{
	// Download the depth jacobian
	std::vector<unsigned short> knn_patch_array;
	std::vector<float> knn_patch_spatial_weight_array;
	std::vector<float> knn_patch_connect_weight_array;
	std::vector<GradientOfDenseImage> gradient_array;
	term2jacobian.knn_patch_array.Download(knn_patch_array);
	term2jacobian.knn_patch_spatial_weight_array.Download(knn_patch_spatial_weight_array);
	term2jacobian.knn_patch_connect_weight_array.Download(knn_patch_connect_weight_array);
	term2jacobian.gradient_array.Download(gradient_array);

	// Simple sanity check
	STAR_CHECK_EQ(knn_patch_array.size(), knn_patch_spatial_weight_array.size());
	STAR_CHECK_EQ(knn_patch_array.size(), gradient_array.size() * d_surfel_knn_size);

	// Iterate through costs
	for (auto i = 0; i < knn_patch_array.size() / d_surfel_knn_size; i++)
	{
		const unsigned short *knn_patch = (unsigned short *)(&knn_patch_array[size_t(i) * d_surfel_knn_size]);
		const float *knn_patch_spatial_weight = (const float *)(&knn_patch_spatial_weight_array[size_t(i) * d_surfel_knn_size]);
		const float *knn_patch_connect_weight = (const float *)(&knn_patch_connect_weight_array[size_t(i) * d_surfel_knn_size]);
		for (auto j = 0; j < d_surfel_knn_size; j++)
		{
			unsigned short node_idx = knn_patch[j];
			float weight = knn_patch_spatial_weight[j] * knn_patch_connect_weight[j];
			float weight_square = weight * weight;
			float *jtj = &jtj_flatten[size_t(node_idx) * d_node_variable_dim_square];
			for (auto channel = 0; channel < d_dense_image_residual_dim; ++channel)
			{
				const float *jacobian = (float *)&(gradient_array[i].gradient[channel]);
				for (int jac_row = 0; jac_row < d_node_variable_dim; jac_row++)
				{
					for (int jac_col = 0; jac_col < d_node_variable_dim; jac_col++)
					{
						jtj[d_node_variable_dim * jac_row + jac_col] += weight_square_vec[channel] * weight_square * jacobian[jac_col] * jacobian[jac_row];

						if (verbose)
						{
							size_t element_idx = size_t(node_idx) * d_node_variable_dim_square + size_t(d_node_variable_dim) * jac_row + jac_col;
							if (element_idx == inspect_index)
							{
								printf("[%d, %d], CPU: DenseImage Term: %d, val: %f, jac at node_%d: (%d, %d, %d), weight: (%f, %f, %f), Jac : (%f, %f, %f, %f, %f, %f).\n",
									   node_idx * d_node_variable_dim_square + d_node_variable_dim * jac_row + jac_col, int(jtj_flatten.size()),
									   i,
									   jtj[d_node_variable_dim * jac_row + jac_col],
									   node_idx, jac_row, jac_col, channel,
									   weight_square_vec[channel], knn_patch_spatial_weight[j], knn_patch_connect_weight[j],
									   jacobian[0], jacobian[1], jacobian[2],
									   jacobian[3], jacobian[4], jacobian[5]);
							}
						}
					}
				}
			}
		}
	}
}

/** Non-Diagonal Jacoba in Debug
 */
void star::updateDenseImageJtJBlockHost(
	const GArrayView<unsigned> &encoded_nodepair,
	std::vector<float> &jtj_flatten,
	const DenseImageTerm2Jacobian &term2jacobian,
	const floatX<d_dense_image_residual_dim> &term_weight_square_vec)
{
	const auto num_nodepairs = encoded_nodepair.Size();
	STAR_CHECK_EQ(num_nodepairs * d_node_variable_dim_square, jtj_flatten.size());

	// Download the jacobian
	std::vector<unsigned short> knn_patch_array;
	std::vector<float> knn_patch_spatial_weight_array;
	std::vector<float> knn_patch_connect_weight_array;
	std::vector<GradientOfDenseImage> gradient_array;
	term2jacobian.knn_patch_array.Download(knn_patch_array);
	term2jacobian.knn_patch_spatial_weight_array.Download(knn_patch_spatial_weight_array);
	term2jacobian.knn_patch_connect_weight_array.Download(knn_patch_connect_weight_array);
	term2jacobian.gradient_array.Download(gradient_array);

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

	// Simple sanity check
	STAR_CHECK_EQ(knn_patch_array.size(), knn_patch_spatial_weight_array.size());
	STAR_CHECK_EQ(knn_patch_array.size(), knn_patch_connect_weight_array.size());
	STAR_CHECK_EQ(knn_patch_array.size(), gradient_array.size() * d_surfel_knn_size);

	// Iterate through costs
	for (auto term_idx = 0; term_idx < gradient_array.size(); term_idx++)
	{
		const unsigned short *knn = (unsigned short *)(&knn_patch_array[size_t(term_idx) * d_surfel_knn_size]);
		const float *knn_patch_spatial_weight = (const float *)(&knn_patch_spatial_weight_array[size_t(term_idx) * d_surfel_knn_size]);
		const float *knn_patch_connect_weight = (const float *)(&knn_patch_connect_weight_array[size_t(term_idx) * d_surfel_knn_size]);

		for (auto i = 0; i < d_surfel_knn_size; i++)
		{
			unsigned short node_i = knn[i];
			float weight_i = knn_patch_spatial_weight[i] * knn_patch_connect_weight[i];
			for (auto j = 0; j < d_surfel_knn_size; j++)
			{
				if (i == j)
					continue;
				unsigned short node_j = knn[j];
				float weight_j = knn_patch_spatial_weight[j] * knn_patch_connect_weight[j];
				auto encoded_ij = encode_nodepair(node_i, node_j);
				auto iter = pair2index.find(encoded_ij);
				if (iter != pair2index.end())
				{
					unsigned index = iter->second;
					for (auto channel = 0; channel < d_dense_image_residual_dim; ++channel)
					{
						const float *jacobian = (float *)&(gradient_array[term_idx].gradient[channel]);
						float *jtj = &jtj_flatten[size_t(index) * d_node_variable_dim_square];
						for (int jac_row = 0; jac_row < d_node_variable_dim; jac_row++)
						{
							for (int jac_col = 0; jac_col < d_node_variable_dim; jac_col++)
							{
								jtj[d_node_variable_dim * jac_row + jac_col] += term_weight_square_vec[channel] * weight_i * weight_j * jacobian[jac_col] * jacobian[jac_row];
							}
						}
					}
				}
				else
				{
					LOG(FATAL) << "Cannot find the index for node " << node_i << " and " << node_j << " pair in term: " << term_idx << ".";
				}
			}
		}
	}
}

/** JtResidual
 */
void star::updateDenseImageJtResidualHost(
	std::vector<float> &jt_residual,
	const DenseImageTerm2Jacobian &term2jacobian,
	const floatX<d_dense_image_residual_dim> &term_weight_square_vec)
{

	// Download the data
	std::vector<unsigned short> knn_patch_array;
	std::vector<float> knn_patch_spatial_weight_array;
	std::vector<float> knn_patch_connect_weight_array;
	std::vector<GradientOfDenseImage> gradient_array;
	std::vector<floatX<d_dense_image_residual_dim>> residual_array;
	term2jacobian.knn_patch_array.Download(knn_patch_array);
	term2jacobian.knn_patch_spatial_weight_array.Download(knn_patch_spatial_weight_array);
	term2jacobian.knn_patch_connect_weight_array.Download(knn_patch_connect_weight_array);
	term2jacobian.gradient_array.Download(gradient_array);
	term2jacobian.residual_array.Download(residual_array);

	// Iterates over terms
	for (auto i = 0; i < residual_array.size(); i++)
	{
		const unsigned short *knn = (unsigned short *)(&knn_patch_array[size_t(i) * d_surfel_knn_size]);
		const float *knn_patch_spatial_weight = (const float *)(&knn_patch_spatial_weight_array[size_t(i) * d_surfel_knn_size]);
		const float *knn_patch_connect_weight = (const float *)(&knn_patch_connect_weight_array[size_t(i) * d_surfel_knn_size]);
		for (auto j = 0; j < d_surfel_knn_size; j++)
		{
			unsigned short node_idx = knn[j];
			float weight = knn_patch_connect_weight[j] * knn_patch_spatial_weight[j];
			float weight_square = weight * weight;
			float *jt_r_node = &jt_residual[size_t(node_idx) * d_node_variable_dim];
			for (auto channel = 0; channel < d_dense_image_residual_dim; ++channel)
			{
				const float *jacobian = (float *)&(gradient_array[i].gradient[channel]);
				const float residual = residual_array[i][channel];
				for (auto k = 0; k < d_node_variable_dim; k++)
				{
					jt_r_node[k] += -term_weight_square_vec[channel] * weight_square * residual * jacobian[k];
				}
			}
		}
	}
}
