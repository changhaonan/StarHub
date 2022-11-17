#include <mono_star/opt/DenseImageHandler.h>

star::DenseImageHandler::DenseImageHandler()
{
	const auto &config = ConfigParser::Instance();
	m_num_cam = config.num_cam();
	for (auto cam_idx = 0; cam_idx < m_num_cam; ++cam_idx)
	{
		m_image_height[cam_idx] = config.downsample_img_rows(cam_idx);
		m_image_width[cam_idx] = config.downsample_img_cols(cam_idx);
		m_project_intrinsic[cam_idx] = config.rgb_intrinsic_downsample(cam_idx);
	}
	memset(&m_depth_observation, 0, sizeof(m_depth_observation));
	memset(&m_geometry_maps, 0, sizeof(m_geometry_maps));
}

void star::DenseImageHandler::AllocateBuffer()
{
	unsigned num_pixels_merge = 0;
	for (auto cam_idx = 0; cam_idx < m_num_cam; ++cam_idx)
	{
		const auto num_pixels = m_image_height[cam_idx] * m_image_width[cam_idx];
		// The buffer for gradient
		m_term_residual[cam_idx].AllocateBuffer(num_pixels);
		m_term_gradient[cam_idx].AllocateBuffer(num_pixels);

		// The buffer for alignment error
		createFloat1TextureSurface(m_image_height[cam_idx], m_image_width[cam_idx], m_alignment_error_map[cam_idx]); // FIXME: when is this being released/unregistered?
		num_pixels_merge += num_pixels;
	}
	m_term_residual_merge.AllocateBuffer(num_pixels_merge);
	m_term_gradient_merge.AllocateBuffer(num_pixels_merge);

	m_node_accumulate_error.AllocateBuffer(Constants::kMaxNumNodes);
	m_node_accumulate_weight.AllocateBuffer(Constants::kMaxNumNodes);
	m_node_normalized_error.AllocateBuffer(Constants::kMaxNumNodes);
	// Outlier
	m_node_outlier_status.AllocateBuffer(Constants::kMaxNumNodes);
}

void star::DenseImageHandler::ReleaseBuffer()
{
	for (auto cam_idx = 0; cam_idx < m_num_cam; ++cam_idx)
	{
		m_term_residual[cam_idx].ReleaseBuffer();
		m_term_gradient[cam_idx].ReleaseBuffer();
	}
	m_term_residual_merge.ReleaseBuffer();
	m_term_gradient_merge.ReleaseBuffer();

	m_node_accumulate_error.ReleaseBuffer();
	m_node_accumulate_weight.ReleaseBuffer();
	m_node_normalized_error.ReleaseBuffer();
	// Outlier
	m_node_outlier_status.ReleaseBuffer();
}

void star::DenseImageHandler::SetInputs(
	// The potential pixels,
	const ImageTermKNNFetcher::ImageTermPixelAndKNN &pixels_knn,
	// Input from other components
	const GArray2D<KNNAndWeight<d_surfel_knn_size>> *knn_map,
	const Measure4Solver &measure4solver,
	const Render4Solver &render4solver,
	const OpticalFlow4Solver &opticalflow4solver,
	const Extrinsic *world2camera)
{
	m_potential_pixels_knn = pixels_knn;

	for (auto cam_idx = 0; cam_idx < m_num_cam; ++cam_idx)
	{
		m_knn_map[cam_idx] = knn_map[cam_idx];

		// Depth observation
		m_depth_observation.vertex_map[cam_idx] = measure4solver.vertex_confid_map[cam_idx];
		m_depth_observation.normal_map[cam_idx] = measure4solver.normal_radius_map[cam_idx];
		// Render map
		m_geometry_maps.reference_vertex_map[cam_idx] = render4solver.reference_vertex_map[cam_idx];
		m_geometry_maps.reference_normal_map[cam_idx] = render4solver.reference_normal_map[cam_idx];
		m_geometry_maps.index_map[cam_idx] = render4solver.index_map[cam_idx];
		// Opticalflow map, attached to geometry map
		m_geometry_maps.opticalflow_map[cam_idx] = opticalflow4solver.opticalflow_map[cam_idx];

		mat34 world2camera_mat = world2camera[cam_idx];
		m_world2cam[cam_idx] = world2camera_mat;
		m_cam2world[cam_idx] = world2camera_mat.inverse();
	}
}

void star::DenseImageHandler::UpdateInputs(
	const GArrayView<DualQuaternion> &node_se3,
	const ImageTermKNNFetcher::ImageTermPixelAndKNN &pixels_knn)
{
	m_node_se3 = node_se3;
	m_potential_pixels_knn = pixels_knn;
}

void star::DenseImageHandler::UpdateNodeSE3(star::GArrayView<star::DualQuaternion> node_se3)
{
	STAR_CHECK_EQ(node_se3.Size(), m_node_se3.Size());
	m_node_se3 = node_se3;
}