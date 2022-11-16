#pragma once
#include <star/opt/solver_types.h>
#include <star/opt/solver_encode.h>

namespace star::device
{

	/** JtJ Diagonal-related
	 */
	__device__ __forceinline__ void computeDenseImageJtJDiagonalJacobian(
		const DenseImageTerm2Jacobian &term2jacobian,
		unsigned node_idx, unsigned typed_term_idx,
		float *__restrict__ jacobian_channelled,
		float &weight)
	{
		unsigned offset;
		const unsigned knn_patch_offset = typed_term_idx * d_surfel_knn_size;
		for (auto i = 0; i < d_surfel_knn_size; ++i)
		{
			if (node_idx == term2jacobian.knn_patch_array[size_t(knn_patch_offset) + i])
			{
				offset = i;
			}
		}
		const float spatial_weight = term2jacobian.knn_patch_spatial_weight_array[size_t(knn_patch_offset) + offset];
		const float connect_weight = term2jacobian.knn_patch_connect_weight_array[size_t(knn_patch_offset) + offset];
		weight = spatial_weight * connect_weight;

		// Jacobian of transform part
		GradientOfDenseImage *gradient = (GradientOfDenseImage *)jacobian_channelled;
		GradientOfDenseImage twist_gradient = term2jacobian.gradient_array[typed_term_idx];
#pragma unroll
		for (auto channel = 0; channel < d_dense_image_residual_dim; ++channel)
		{
			GradientOfScalarCost *gradient = (GradientOfScalarCost *)(jacobian_channelled + channel * size_t(d_node_variable_dim));
			gradient->rotation.x = twist_gradient.gradient[channel].rotation.x * weight;
			gradient->rotation.y = twist_gradient.gradient[channel].rotation.y * weight;
			gradient->rotation.z = twist_gradient.gradient[channel].rotation.z * weight;
			gradient->translation.x = twist_gradient.gradient[channel].translation.x * weight;
			gradient->translation.y = twist_gradient.gradient[channel].translation.y * weight;
			gradient->translation.z = twist_gradient.gradient[channel].translation.z * weight;
		}
	}

	/** JtResidual-related
	 */
	__device__ __forceinline__ void computeDenseImageJacobianTransposeDot(
		const DenseImageTerm2Jacobian &term2jacobian,
		unsigned node_idx, unsigned typed_term_idx,
		float *__restrict__ jt_residual,
		const floatX<d_dense_image_residual_dim> &opt_weight_square_vec)
	{
		const floatX<d_dense_image_residual_dim> residual_value = term2jacobian.residual_array[typed_term_idx];
		unsigned offset;
		const unsigned knn_patch_offset = typed_term_idx * d_surfel_knn_size;
		for (auto i = 0; i < d_surfel_knn_size; ++i)
		{
			if (node_idx == term2jacobian.knn_patch_array[size_t(knn_patch_offset) + i])
			{
				offset = i;
			}
		}
		const float spatial_weight = term2jacobian.knn_patch_spatial_weight_array[size_t(knn_patch_offset) + offset];
		const float connect_weight = term2jacobian.knn_patch_connect_weight_array[size_t(knn_patch_offset) + offset];
		const float weight = spatial_weight * connect_weight;
		const float weight_square = weight * weight;
		GradientOfDenseImage twist_gradient = term2jacobian.gradient_array[typed_term_idx];

#pragma unroll
		for (auto channel = 0; channel < d_dense_image_residual_dim; ++channel)
		{
			float opt_weigth_square = opt_weight_square_vec[channel] * weight_square;
			for (auto i = 0; i < d_node_variable_dim; i++)
			{
				float *gradient_array = (float *)(&twist_gradient.gradient[channel]);
				jt_residual[i] += opt_weigth_square * residual_value[channel] * gradient_array[i];
			}
		}
	}

	/** JtJ non-Diagonal-related
	 */
	__device__ __forceinline__ void computeDenseImageJtJBlockJacobian(
		const DenseImageTerm2Jacobian &term2jacobian,
		unsigned encoded_pair, unsigned typed_term_idx,
		float *__restrict__ jacobian_channelled,
		float *weight)
	{
		unsigned node_i, node_j;
		decode_nodepair(encoded_pair, node_i, node_j);

		// Weight
		unsigned offset_i, offset_j;
		const unsigned knn_patch_offset = typed_term_idx * d_surfel_knn_size;
		for (auto i = 0; i < d_surfel_knn_size; ++i)
		{
			if (node_i == term2jacobian.knn_patch_array[size_t(knn_patch_offset) + i])
			{
				offset_i = i;
			}
			if (node_j == term2jacobian.knn_patch_array[size_t(knn_patch_offset) + i])
			{
				offset_j = i;
			}
		}
		const float spatial_weight_i = term2jacobian.knn_patch_spatial_weight_array[size_t(knn_patch_offset) + offset_i];
		const float connect_weight_i = term2jacobian.knn_patch_connect_weight_array[size_t(knn_patch_offset) + offset_i];
		const float weight_i = spatial_weight_i * connect_weight_i;

		const float spatial_weight_j = term2jacobian.knn_patch_spatial_weight_array[size_t(knn_patch_offset) + offset_j];
		const float connect_weight_j = term2jacobian.knn_patch_connect_weight_array[size_t(knn_patch_offset) + offset_j];
		const float weight_j = spatial_weight_j * connect_weight_j;

		*weight = weight_i * weight_j;
		GradientOfDenseImage twist_gradient = term2jacobian.gradient_array[typed_term_idx];
#pragma unroll
		for (auto channel = 0; channel < d_dense_image_residual_dim; ++channel)
		{
			GradientOfScalarCost *gradient = (GradientOfScalarCost *)(jacobian_channelled + channel * size_t(d_node_variable_dim));
			gradient->rotation.x = twist_gradient.gradient[channel].rotation.x;
			gradient->rotation.y = twist_gradient.gradient[channel].rotation.y;
			gradient->rotation.z = twist_gradient.gradient[channel].rotation.z;
			gradient->translation.x = twist_gradient.gradient[channel].translation.x;
			gradient->translation.y = twist_gradient.gradient[channel].translation.y;
			gradient->translation.z = twist_gradient.gradient[channel].translation.z;
		}
	}
}
