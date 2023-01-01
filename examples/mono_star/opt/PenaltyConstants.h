#pragma once
#include <vector_types.h>
#include <mono_star/common/global_configs.h>
#include <star/common/types/typeX.h>

namespace star
{
	// Forward declare
	class SolverIterationData;

	class PenaltyConstants
	{
	private:
		float m_lambda_dense_image_depth; // Depth loss
		float m_lambda_dense_image_optical_flow;
		float m_lambda_reg;				 // Asap loss
		float m_lambda_node_translation; // Node motion prediction
		float m_lambda_feature;			 // Feature loss

		// Only modifiable by warp solver
		friend class SolverIterationData;
		void __host__ __device__ setDefaultValue();
		void setGlobalIterationValue();
		void setLocalIterationValue();

	public:
		__host__ __device__ PenaltyConstants();

		// All access other than WarpSolver should be read-only
		__host__ __device__ __forceinline__ float DenseImageDepth() const { return m_lambda_dense_image_depth; }
		__host__ __device__ __forceinline__ float DenseImageDepthSquared() const { return m_lambda_dense_image_depth * m_lambda_dense_image_depth; }
		__host__ __device__ __forceinline__ float DenseImageOpticalFlow() const { return m_lambda_dense_image_optical_flow; }
		__host__ __device__ __forceinline__ float DenseImageOpticalFlowSquared() const { return m_lambda_dense_image_optical_flow * m_lambda_dense_image_optical_flow; }
		__host__ __device__ __forceinline__ floatX<d_dense_image_residual_dim> DenseImageVec() const
		{
			return make_floatX<d_dense_image_residual_dim>({m_lambda_dense_image_depth,
															m_lambda_dense_image_optical_flow,
															m_lambda_dense_image_optical_flow});
		}
		__host__ __device__ __forceinline__ floatX<d_dense_image_residual_dim> DenseImageSquaredVec() const
		{
			return make_floatX<d_dense_image_residual_dim>({m_lambda_dense_image_depth * m_lambda_dense_image_depth,
															m_lambda_dense_image_optical_flow * m_lambda_dense_image_optical_flow,
															m_lambda_dense_image_optical_flow * m_lambda_dense_image_optical_flow});
		}

		__host__ __device__ __forceinline__ float Reg() const { return m_lambda_reg; }
		__host__ __device__ __forceinline__ float RegSquared() const { return m_lambda_reg * m_lambda_reg; }

		__host__ __device__ __forceinline__ float NodeTranslation() const { return m_lambda_node_translation; }
		__host__ __device__ __forceinline__ float NodeTranslationSquared() const { return m_lambda_node_translation * m_lambda_node_translation; }

		__host__ __device__ __forceinline__ float Feature() const { return m_lambda_feature; }
		__host__ __device__ __forceinline__ float FeatureSquared() const { return m_lambda_feature; }
	};

}