#include "MeasureProcessorOffline.h"

star::MeasureProcessorOffline::MeasureProcessorOffline()
{
	auto &config = ConfigParser::Instance();

	m_fetcher = std::make_shared<VolumeDeformFileFetch>(config.data_path());
	m_surfel_map = std::make_shared<SurfelMap>(config.downsample_img_cols(), config.downsample_img_rows());
	m_surfel_map_initializer = std::make_shared<SurfelMapInitializer>(
		config.downsample_img_cols(),
		config.downsample_img_rows(),
		config.clip_near(),
		config.clip_far(),
		config.surfel_radius_scale(),
		config.rgb_intrinsic_downsample());

	m_start_frame_idx = config.start_frame_idx();
	m_step_frame = config.step_frame();

	// Allocate buffer
	size_t num_pixel = config.downsample_img_cols() * config.downsample_img_rows();
	m_g_color_img.create(num_pixel);
	m_g_depth_img.create(num_pixel);
}

void star::MeasureProcessorOffline::Process(
	StarStageBuffer &star_stage_buffer_this,
	const StarStageBuffer &star_stage_buffer_prev,
	cudaStream_t stream,
	const unsigned frame_idx)
{

	// Do nothing now
}

void star::MeasureProcessorOffline::processFrame(
	const unsigned frame_idx,
	cudaStream_t stream)
{
	// 1. Load a rgb image && depth image
	const auto image_idx = size_t(frame_idx) * m_step_frame + m_start_frame_idx;
	m_fetcher->FetchRGBImage(0, image_idx, m_color_img);
	m_fetcher->FetchDepthImage(0, image_idx, m_depth_img);

	// Synced
	m_g_color_img.upload(m_color_img.ptr<uchar3>(), m_g_color_img.size());
	m_g_depth_img.upload(m_depth_img.ptr<unsigned short>(), m_g_depth_img.size());

	// 2. Initialize surfel map
	m_surfel_map_initializer->InitFromRGBDImage(
		GArrayView(m_g_color_img),
		GArrayView(m_g_depth_img),
		frame_idx,
		*m_surfel_map,
		stream);
	cudaSafeCall(cudaStreamSynchronize(stream));

	// 3. Visualize
	saveContext(frame_idx, stream);
}

void star::MeasureProcessorOffline::saveContext(
	const unsigned frame_idx,
	cudaStream_t stream)
{
	auto& context = easy3d::Context::Instance();
	context.open(frame_idx);
	context.addPointCloud("measure");
	visualize::SavePointCloud(m_surfel_map->VertexConfigReadOnly(), context.at("measure"));
	context.close();
}