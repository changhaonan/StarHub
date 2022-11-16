#pragma once

namespace star::device
{

	/** JtJ Diagonal-related
	 */
	__device__ __forceinline__ void computeNodeTranslationJtJDiagonalJacobian(
		const NodeTranslationTerm2Jacobian &term2jacobian,
		unsigned node_idx, unsigned typed_term_idx,
		float *__restrict__ channelled_jacobian)
	{
		const float pred_confid = term2jacobian.node_motion_pred[typed_term_idx].w;
		GradientOfScalarCost *gradient = (GradientOfScalarCost *)channelled_jacobian;
		gradient[0].rotation = make_float3(0.0f, 0.0f, 0.0f) * pred_confid;
		gradient[0].translation = make_float3(1.0f, 0.0f, 0.0f) * pred_confid;
		gradient[1].rotation = make_float3(0.0f, 0.0f, 0.0f) * pred_confid;
		gradient[1].translation = make_float3(0.0f, 1.0f, 0.0f) * pred_confid;
		gradient[2].rotation = make_float3(0.0f, 0.0f, 0.0f) * pred_confid;
		gradient[2].translation = make_float3(0.0f, 0.0f, 1.0f) * pred_confid;
	}

	/** JtResidual-related
	 */
	__device__ __forceinline__ void computeNodeMotionJtResidual(
		const NodeTranslationTerm2Jacobian &term2jacobian,
		unsigned node_idx, unsigned typed_term_idx,
		float *__restrict__ jt_residual)
	{
		const float4 node_motion_pred = term2jacobian.node_motion_pred[typed_term_idx];
		const float3 T_translation = term2jacobian.T_translation[typed_term_idx];
		const float pred_confid_square = node_motion_pred.w * node_motion_pred.w;
		// Only influence translation term
		jt_residual[3] = (T_translation.x - node_motion_pred.x) * pred_confid_square;
		jt_residual[4] = (T_translation.y - node_motion_pred.y) * pred_confid_square;
		jt_residual[5] = (T_translation.z - node_motion_pred.z) * pred_confid_square;
	}
}