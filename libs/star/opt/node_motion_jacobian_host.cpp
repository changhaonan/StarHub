#include "node_motion_jacobian_host.h"

/** Diagonal Jacobian
*/
void star::updateNodeTranslationJtJDiagonalHost(
	std::vector<float>& jtj_flatten,
	const NodeTranslationTerm2Jacobian& node_translation_term2jacobian,
	const float weight_square) {
	std::vector<float4> node_motion_pred;
	std::vector<float3> T_translation;
	node_translation_term2jacobian.node_motion_pred.Download(node_motion_pred);
	node_translation_term2jacobian.T_translation.Download(T_translation);
	for (auto i = 0; i < node_motion_pred.size(); i++) {
		const float pred_confid = node_motion_pred[i].w;
		float channelled_jacobian[d_node_variable_dim * 3] = { 0 };
		GradientOfScalarCost* gradient = (GradientOfScalarCost*)channelled_jacobian;
		gradient[0].rotation = make_float3(0.0f, 0.0f, 0.0f) * pred_confid;
		gradient[0].translation = make_float3(1.0f, 0.0f, 0.0f) * pred_confid;
		gradient[1].rotation = make_float3(0.0f, 0.0f, 0.0f) * pred_confid;
		gradient[1].translation = make_float3(0.0f, 1.0f, 0.0f) * pred_confid;
		gradient[2].rotation = make_float3(0.0f, 0.0f, 0.0f) * pred_confid;
		gradient[2].translation = make_float3(0.0f, 0.0f, 1.0f) * pred_confid;

		float residual[3] = { 0 };  // Lacking here
		float* jtj = &jtj_flatten[size_t(i) * d_node_variable_dim_square];
		for (auto channel = 0; channel < 3; channel++) {
			float* jacobian = (float*)(&channelled_jacobian[channel * d_node_variable_dim]);
			for (int jac_row = 0; jac_row < d_node_variable_dim; jac_row++) {
				for (int jac_col = 0; jac_col < d_node_variable_dim; jac_col++) {
					jtj[d_node_variable_dim * jac_row + jac_col] += weight_square * jacobian[jac_col] * jacobian[jac_row];
					////[Debug]
					//if ((i * d_node_variable_dim_square + d_node_variable_dim * jac_row + jac_col) == 0) {
					//	printf("CPU: term: %d, val: %f, node jacobian at node_%d: (%d, %d) is : (%f, %f, %f, %f, %f, %f).\n",
					//		i,
					//		jtj[d_node_variable_dim * jac_row + jac_col],
					//		i, jac_row, jac_col,
					//		jacobian[0], jacobian[1], jacobian[2],
					//		jacobian[3], jacobian[4], jacobian[5]);
					//}
				}
			}
		}
	}
}

/** JtResidual
*/
void star::updateNodeTranslationJtResidualHost(
	std::vector<float>& jt_residual,
	const NodeTranslationTerm2Jacobian& node_translation_term2jacobian,
	const float weight_square) {
	std::vector<float4> node_motion_pred_array;
	std::vector<float3> T_translation_array;

	node_translation_term2jacobian.node_motion_pred.Download(node_motion_pred_array);
	node_translation_term2jacobian.T_translation.Download(T_translation_array);

	for (auto i = 0; i < node_motion_pred_array.size(); ++i) {
		const float pred_confid_square = node_motion_pred_array[i].w * node_motion_pred_array[i].w;
		// Only influence translation term
		jt_residual[size_t(i) * d_node_variable_dim + 3] += -weight_square * (T_translation_array[i].x - node_motion_pred_array[i].x) * pred_confid_square;
		jt_residual[size_t(i) * d_node_variable_dim + 4] += -weight_square * (T_translation_array[i].y - node_motion_pred_array[i].y) * pred_confid_square;
		jt_residual[size_t(i) * d_node_variable_dim + 5] += -weight_square * (T_translation_array[i].z - node_motion_pred_array[i].z) * pred_confid_square;
	}
}