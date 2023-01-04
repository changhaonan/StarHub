#include <iostream>
#include <mono_star/opt/DenseImageHandler.h>
#include <mono_star/opt/PenaltyConstants.h>

float star::DenseImageHandler::computeSOR() const
{
	// 1. Prepare
	std::vector<floatX<d_dense_image_residual_dim>> h_residual_pixel;
	m_term_residual_merge.View().Download(h_residual_pixel);
	float residual_depth_sum = 0.f;
	float residual_opticalflow_sum = 0.f;
	auto penalty = PenaltyConstants();

	// 2. Sum
	for (auto i = 0; i < h_residual_pixel.size(); ++i)
	{
		// 2.1. Depth
		residual_depth_sum += h_residual_pixel[i][0] * h_residual_pixel[i][0] * penalty.DenseImageDepthSquared();
		// 2.2. Opticalflow
		residual_opticalflow_sum += h_residual_pixel[i][1] * h_residual_pixel[i][1] * penalty.DenseImageOpticalFlowSquared();
		residual_opticalflow_sum += h_residual_pixel[i][2] * h_residual_pixel[i][2] * penalty.DenseImageOpticalFlowSquared();
	}

	// 3. Log
	std::cout << "SOR [Depth]: " << residual_depth_sum << std::endl;
	std::cout << "SOR [OF]: " << residual_opticalflow_sum << std::endl;
	return residual_depth_sum + residual_opticalflow_sum;
}

void star::DenseImageHandler::jacobianTermCheck()
{
	// Sanity size check
	for (auto cam_idx = 0; cam_idx < m_num_cam; ++cam_idx)
	{
		STAR_CHECK_EQ(m_term_residual[cam_idx].ArraySize(), m_potential_pixels_knn.pixels[cam_idx].Size());
		STAR_CHECK_EQ(m_term_gradient[cam_idx].ArraySize(), m_potential_pixels_knn.pixels[cam_idx].Size());
	}

	// Term sanity
	for (auto cam_idx = 0; cam_idx < m_num_cam; ++cam_idx)
	{
		std::vector<floatX<d_dense_image_residual_dim>> h_term_residual;
		m_term_residual[cam_idx].View().Download(h_term_residual);

		// for (float residual : h_term_residual)
		//	std::cout << residual << std::endl;
	}
}