#pragma once
#include <star/common/common_types.h>
#include <star/math/vector_ops.hpp>
#include <star/math/DualQuaternion.hpp>
#include <star/opt/solver_types.h>
#include <star/opt/huber_weight.h>

namespace star::device
{

	/**
	 * \brief The method to compute the jacobian and residual of node graph reg term.
	 *        The underlying form is the same, the variants are for efficiency
	 */
	__host__ __forceinline__ void computeRegTermJacobianResidual(
		const float3 &Ti_xj, const float3 &Tj_xj,
		const float connect_weight,
		float *residual,			   // [3]
		GradientOfRegCost *gradient_i, //[3]
		GradientOfRegCost *gradient_j  //[3]
	)
	{
		// Compute the residual
		residual[0] = (Ti_xj.x - Tj_xj.x) * connect_weight;
		residual[1] = (Ti_xj.y - Tj_xj.y) * connect_weight;
		residual[2] = (Ti_xj.z - Tj_xj.z) * connect_weight;

		// Compute the jacobian
		gradient_i[0].rotation = make_float3(0.0f, Ti_xj.z, -Ti_xj.y) * connect_weight;
		gradient_i[1].rotation = make_float3(-Ti_xj.z, 0.0f, Ti_xj.x) * connect_weight;
		gradient_i[2].rotation = make_float3(Ti_xj.y, -Ti_xj.x, 0.0f) * connect_weight;
		gradient_i[0].translation = make_float3(1.0f, 0.0f, 0.0f) * connect_weight;
		gradient_i[1].translation = make_float3(0.0f, 1.0f, 0.0f) * connect_weight;
		gradient_i[2].translation = make_float3(0.0f, 0.0f, 1.0f) * connect_weight;

		gradient_j[0].rotation = make_float3(0.0f, -Tj_xj.z, Tj_xj.y) * connect_weight;
		gradient_j[1].rotation = make_float3(Tj_xj.z, 0.0f, -Tj_xj.x) * connect_weight;
		gradient_j[2].rotation = make_float3(-Tj_xj.y, Tj_xj.x, 0.0f) * connect_weight;
		gradient_j[0].translation = make_float3(-1.0f, 0.0f, 0.0f) * connect_weight;
		gradient_j[1].translation = make_float3(0.0f, -1.0f, 0.0f) * connect_weight;
		gradient_j[2].translation = make_float3(0.0f, 0.0f, -1.0f) * connect_weight;
	}

	__host__ __forceinline__ void computeRegTermJacobianResidual(
		const float4 &xi4, const float4 &xj4,
		const mat34 &Ti, const mat34 &Tj,
		const float connect_weight,
		float *residual,			   // [3]
		GradientOfRegCost *gradient_i, //[3]
		GradientOfRegCost *gradient_j  //[3]
	)
	{
		const float3 xi = make_float3(xi4.x, xi4.y, xi4.z);
		const float3 xj = make_float3(xj4.x, xj4.y, xj4.z);
		const float3 r = Ti.rot * xj + Ti.trans;
		const float3 s = Tj.rot * xj + Tj.trans;

		computeRegTermJacobianResidual(r, s, connect_weight, residual, gradient_i, gradient_j);
	}

	__host__ __device__ __forceinline__ void computeRegTermResidual(
		const float3 &Ti_xj,
		const float3 &Tj_xj,
		float *residual // [3]
	)
	{
		residual[0] = Ti_xj.x - Tj_xj.x;
		residual[1] = Ti_xj.y - Tj_xj.y;
		residual[2] = Ti_xj.z - Tj_xj.z;
	}

	__host__ __device__ __forceinline__ void computeRegTermResidual(
		const float4 &xj4,
		const mat34 &Ti, const mat34 &Tj,
		float *residual // [3]
	)
	{
		const float3 xj = make_float3(xj4.x, xj4.y, xj4.z);
		const float3 r = Ti.rot * xj + Ti.trans;
		const float3 s = Tj.rot * xj + Tj.trans;

		// Compute the residual
		computeRegTermResidual(r, s, residual);
	}

	__host__ __device__ __forceinline__ void computeRegTermJacobian(
		const float3 &Ti_xj,
		const float3 &Tj_xj,
		GradientOfRegCost *gradient_i, //[3]
		GradientOfRegCost *gradient_j  //[3]
	)
	{
		// Compute the jacobian
		gradient_i[0].rotation = make_float3(0.0f, Ti_xj.z, -Ti_xj.y);
		gradient_i[0].translation = make_float3(1.0f, 0.0f, 0.0f);
		gradient_i[1].rotation = make_float3(-Ti_xj.z, 0.0f, Ti_xj.x);
		gradient_i[1].translation = make_float3(0.0f, 1.0f, 0.0f);
		gradient_i[2].rotation = make_float3(Ti_xj.y, -Ti_xj.x, 0.0f);
		gradient_i[2].translation = make_float3(0.0f, 0.0f, 1.0f);

		gradient_j[0].rotation = make_float3(0.0f, -Tj_xj.z, Tj_xj.y);
		gradient_j[0].translation = make_float3(-1.0f, 0.0f, 0.0f);
		gradient_j[1].rotation = make_float3(Tj_xj.z, 0.0f, -Tj_xj.x);
		gradient_j[1].translation = make_float3(0.0f, -1.0f, 0.0f);
		gradient_j[2].rotation = make_float3(-Tj_xj.y, Tj_xj.x, 0.0f);
		gradient_j[2].translation = make_float3(0.0f, 0.0f, -1.0f);
	}

	__host__ __device__ __forceinline__ void computeRegTermJacobian(
		const float4 &xj4,
		const mat34 &Ti, const mat34 &Tj,
		GradientOfRegCost *gradient_i, //[3]
		GradientOfRegCost *gradient_j  //[3]
	)
	{
		const float3 xj = make_float3(xj4.x, xj4.y, xj4.z);
		const float3 r = Ti.rot * xj + Ti.trans;
		const float3 s = Tj.rot * xj + Tj.trans;

		// Compute the jacobian
		computeRegTermJacobian(r, s, gradient_i, gradient_j);
	}

	__host__ __device__ __forceinline__ void computeRegTermJacobian(
		const float3 &Ti_xj, const float3 &Tj_xj,
		const float connect_weight,
		bool is_node_i,
		GradientOfRegCost *gradient //[3]
	)
	{
		if (is_node_i)
		{
			gradient[0].rotation = make_float3(0.0f, Ti_xj.z, -Ti_xj.y) * connect_weight;
			gradient[0].translation = make_float3(1.0f, 0.0f, 0.0f) * connect_weight;
			gradient[1].rotation = make_float3(-Ti_xj.z, 0.0f, Ti_xj.x) * connect_weight;
			gradient[1].translation = make_float3(0.0f, 1.0f, 0.0f) * connect_weight;
			gradient[2].rotation = make_float3(Ti_xj.y, -Ti_xj.x, 0.0f) * connect_weight;
			gradient[2].translation = make_float3(0.0f, 0.0f, 1.0f) * connect_weight;
		}
		else
		{
			gradient[0].rotation = make_float3(0.0f, -Tj_xj.z, Tj_xj.y) * connect_weight;
			gradient[0].translation = make_float3(-1.0f, 0.0f, 0.0f) * connect_weight;
			gradient[1].rotation = make_float3(Tj_xj.z, 0.0f, -Tj_xj.x) * connect_weight;
			gradient[1].translation = make_float3(0.0f, -1.0f, 0.0f) * connect_weight;
			gradient[2].rotation = make_float3(-Tj_xj.y, Tj_xj.x, 0.0f) * connect_weight;
			gradient[2].translation = make_float3(0.0f, 0.0f, -1.0f) * connect_weight;
		}
	}

	__host__ __device__ __forceinline__ void computeRegTermJacobian(
		const float4 &xj4,
		const mat34 &Ti, const mat34 &Tj,
		const float connect_weight,
		bool is_node_i,
		GradientOfRegCost *gradient //[3]
	)
	{
		const float3 xj = make_float3(xj4.x, xj4.y, xj4.z);
		const float3 r = Ti.rot * xj + Ti.trans;
		const float3 s = Tj.rot * xj + Tj.trans;
		computeRegTermJacobian(r, s, connect_weight, is_node_i, gradient);
	}

	/** JtJ Diagonal related
	 */
	__device__ __forceinline__ void computeRegJtJDiagonalJacobian(
		const NodeGraphRegTerm2Jacobian &term2jacobian,
		unsigned node_idx, unsigned typed_term_idx,
		float *__restrict__ channelled_jacobian)
	{
		const ushort3 node_ij_k = term2jacobian.node_graph[typed_term_idx];
		const auto Ti_xj = term2jacobian.Ti_xj[typed_term_idx];
		const auto Tj_xj = term2jacobian.Tj_xj[typed_term_idx];
		const auto validity = term2jacobian.validity_indicator[typed_term_idx];
		const auto connect_weight = term2jacobian.connect_weight[typed_term_idx];
		const bool is_node_i = (node_idx == node_ij_k.x);
		if (validity == 0)
		{ // Do nothing: left as 0
#pragma unroll
			for (auto i = 0; i < 3 * d_node_variable_dim; i++)
				channelled_jacobian[i] = 0.0f;
			return;
		}
		computeRegTermJacobian(Ti_xj, Tj_xj, connect_weight, is_node_i, (GradientOfScalarCost *)channelled_jacobian);
		if (is_node_i)
		{
			// TODO: do something
		}
	}

	/** JtResidual related
	 */
	__host__ __device__ __forceinline__ void computeRegTermJtResidual(
		const float3 &Ti_xj,
		const float3 &Tj_xj,
		const float connect_weight,
		bool is_node_i,
		float *__restrict__ jt_residual)
	{
		const float connect_weight_square = connect_weight * connect_weight;
		if (is_node_i)
		{
			// First iter: assign
			float residual = (Ti_xj.x - Tj_xj.x) * connect_weight_square;
			*((float3 *)(&jt_residual[0])) = residual * make_float3(0.0f, Ti_xj.z, -Ti_xj.y);
			*((float3 *)(&jt_residual[3])) = residual * make_float3(1.0f, 0.0f, 0.0f);

			// Next iters: plus
			residual = (Ti_xj.y - Tj_xj.y) * connect_weight_square;
			*((float3 *)(&jt_residual[0])) += residual * make_float3(-Ti_xj.z, 0.0f, Ti_xj.x);
			*((float3 *)(&jt_residual[3])) += residual * make_float3(0.0f, 1.0f, 0.0f);

			residual = (Ti_xj.z - Tj_xj.z) * connect_weight_square;
			*((float3 *)(&jt_residual[0])) += residual * make_float3(Ti_xj.y, -Ti_xj.x, 0.0f);
			*((float3 *)(&jt_residual[3])) += residual * make_float3(0.0f, 0.0f, 1.0f);
		}
		else
		{
			// First iter: assign
			float residual = (Ti_xj.x - Tj_xj.x) * connect_weight_square;
			*((float3 *)(&jt_residual[0])) = residual * make_float3(0.0f, -Tj_xj.z, Tj_xj.y);
			*((float3 *)(&jt_residual[3])) = residual * make_float3(-1.0f, 0.0f, 0.0f);

			// Next iters: plus
			residual = (Ti_xj.y - Tj_xj.y) * connect_weight_square;
			*((float3 *)(&jt_residual[0])) += residual * make_float3(Tj_xj.z, 0.0f, -Tj_xj.x);
			*((float3 *)(&jt_residual[3])) += residual * make_float3(0.0f, -1.0f, 0.0f);

			residual = (Ti_xj.z - Tj_xj.z) * connect_weight_square;
			*((float3 *)(&jt_residual[0])) += residual * make_float3(-Tj_xj.y, Tj_xj.x, 0.0f);
			*((float3 *)(&jt_residual[3])) += residual * make_float3(0.0f, 0.0f, -1.0f);
		}
	}

	__host__ __device__ __forceinline__ void computeRegTermJtResidual(
		const float4 &xj4,
		const mat34 &Ti, const mat34 &Tj,
		const float connect_weight,
		bool is_node_i,
		float jt_residual[6])
	{
		const float3 xj = make_float3(xj4.x, xj4.y, xj4.z);
		const float3 r = Ti.rot * xj + Ti.trans;
		const float3 s = Tj.rot * xj + Tj.trans;

		// Combine the computation
		computeRegTermJtResidual(r, s, connect_weight, is_node_i, jt_residual);
	}

	__device__ __forceinline__ void computeRegJtResidual(
		const NodeGraphRegTerm2Jacobian &term2jacobian,
		unsigned node_idx, unsigned typed_term_idx,
		float *__restrict__ jt_residual,
		const unsigned jt_dot_blk_size)
	{
		const ushort3 node_ij_k = term2jacobian.node_graph[typed_term_idx];
		const auto Ti_xj = term2jacobian.Ti_xj[typed_term_idx];
		const auto Tj_xj = term2jacobian.Tj_xj[typed_term_idx];
		const auto validity = term2jacobian.validity_indicator[typed_term_idx];
		const float connect_weight = term2jacobian.connect_weight[typed_term_idx];
		const bool is_node_i = (node_idx == node_ij_k.x);
		if (validity == 0)
		{
			for (auto i = 0; i < jt_dot_blk_size; i++)
				jt_residual[i] = 0.0f;
			return;
		}
		computeRegTermJtResidual(Ti_xj, Tj_xj, connect_weight, is_node_i, jt_residual);
	}

	/** JtJ non-Diagonal-related
	 */
	__device__ __forceinline__ void computeRegJtJLocalBlock(
		const NodeGraphRegTerm2Jacobian &term2jacobian,
		unsigned typed_term_idx,
		unsigned encoded_pair,
		float *__restrict__ local_jtj_blks,
		float weight_square = 1.0f)
	{
		// Check the validity of this term
		const auto validity = term2jacobian.validity_indicator[typed_term_idx];
		if (validity == 0)
		{
#pragma unroll
			for (int jac_idx = 0; jac_idx < d_node_variable_dim_square; jac_idx++)
			{
				local_jtj_blks[jac_idx] = 0.f;
			}
			return;
		}

		// Explicit compute jacobian
		const float3 r = term2jacobian.Ti_xj[typed_term_idx];
		const float3 s = term2jacobian.Tj_xj[typed_term_idx];
		const ushort3 node_ij_k = term2jacobian.node_graph[typed_term_idx];
		const float connect_weight = term2jacobian.connect_weight[typed_term_idx];
		const float connect_weight_square = connect_weight * connect_weight;
		weight_square *= connect_weight_square;
		unsigned node_i, node_j;
		decode_nodepair(encoded_pair, node_i, node_j);

		// The order of two terms
		const float *jacobian_encoded_i;
		const float *jacobian_encoded_j;
		GradientOfScalarCost gradient_i, gradient_j;
		if (node_i == node_ij_k.x)
		{
			jacobian_encoded_i = (const float *)(&gradient_i);
			jacobian_encoded_j = (const float *)(&gradient_j);
		}
		else
		{
			jacobian_encoded_i = (const float *)(&gradient_j);
			jacobian_encoded_j = (const float *)(&gradient_i);
		}

		// The first iteration assign
		gradient_i.rotation = make_float3(0.0f, r.z, -r.y);
		gradient_i.translation = make_float3(1.0f, 0.0f, 0.0f);
		gradient_j.rotation = make_float3(0.0f, -s.z, s.y);
		gradient_j.translation = make_float3(-1.0f, 0.0f, 0.0f);

		for (int jac_row = 0; jac_row < d_node_variable_dim; jac_row++)
		{
#pragma unroll
			for (int jac_col = 0; jac_col < d_node_variable_dim; jac_col++)
			{
				local_jtj_blks[d_node_variable_dim * jac_row + jac_col] = weight_square * jacobian_encoded_i[jac_col] * jacobian_encoded_j[jac_row];
			}
		}

		// The next two iterations, plus
		gradient_i.rotation = make_float3(-r.z, 0.0f, r.x);
		gradient_i.translation = make_float3(0.0f, 1.0f, 0.0f);
		gradient_j.rotation = make_float3(s.z, 0.0f, -s.x);
		gradient_j.translation = make_float3(0.0f, -1.0f, 0.0f);
		for (int jac_row = 0; jac_row < d_node_variable_dim; jac_row++)
		{
#pragma unroll
			for (int jac_col = 0; jac_col < d_node_variable_dim; jac_col++)
			{
				local_jtj_blks[d_node_variable_dim * jac_row + jac_col] += weight_square * jacobian_encoded_i[jac_col] * jacobian_encoded_j[jac_row];
			}
		}

		gradient_i.rotation = make_float3(r.y, -r.x, 0.0f);
		gradient_i.translation = make_float3(0.0f, 0.0f, 1.0f);
		gradient_j.rotation = make_float3(-s.y, s.x, 0.0f);
		gradient_j.translation = make_float3(0.0f, 0.0f, -1.0f);
		for (int jac_row = 0; jac_row < d_node_variable_dim; jac_row++)
		{
#pragma unroll
			for (int jac_col = 0; jac_col < d_node_variable_dim; jac_col++)
			{
				local_jtj_blks[d_node_variable_dim * jac_row + jac_col] += weight_square * jacobian_encoded_i[jac_col] * jacobian_encoded_j[jac_row];
			}
		}
	}
}
