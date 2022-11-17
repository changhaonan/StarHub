#include <mono_star/opt/PenaltyConstants.h>

star::PenaltyConstants::PenaltyConstants()
{
	setDefaultValue();
}

void star::PenaltyConstants::setDefaultValue()
{
	m_lambda_dense_image_depth = 1.0f;
	m_lambda_dense_image_optical_flow = 1.0f;
	m_lambda_reg = 3.0f; // 2.3f for k4, 1.0 for k8
	m_lambda_node_translation = 0.0f;
	m_lambda_feature = 0.f;
}

void star::PenaltyConstants::setGlobalIterationValue()
{
	m_lambda_dense_image_depth = 1.0f;
	m_lambda_dense_image_optical_flow = 1.0f;
	m_lambda_reg = 3.0f; // Global iteration has smaller regulation (0.3)
	m_lambda_node_translation = 0.0f;
	m_lambda_feature = 0.f;
}

void star::PenaltyConstants::setLocalIterationValue()
{
	m_lambda_dense_image_depth = 1.0f;
	m_lambda_dense_image_optical_flow = 1.0f;
	m_lambda_reg = 3.0f; // Local iteration has larger regulation (0.3)
	m_lambda_node_translation = 0.0f;
	m_lambda_feature = 0.f;
}