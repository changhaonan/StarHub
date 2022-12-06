#include "MeasureProcessorOffline.h"
#include <memory>
#include <star/img_proc/image_resize.cuh>
#include <star/visualization/Visualizer.h>

star::MeasureProcessorOffline::MeasureProcessorOffline()
{
	auto &config = ConfigParser::Instance();

	m_fetcher = std::make_shared<VolumeDeformFileFetch>(config.data_path());
	m_surfel_map = std::make_shared<SurfelMap>(config.downsample_img_cols(), config.downsample_img_rows());
	m_surfel_map_prev = std::make_shared<SurfelMap>(config.downsample_img_cols(), config.downsample_img_rows());
	m_surfel_map_initializer = std::make_shared<SurfelMapInitializer>(
		config.raw_img_cols(),
		config.raw_img_rows(),
		config.clip_near(),
		config.clip_far(),
		config.surfel_radius_scale(),
		config.depth_intrinsic_raw());

	m_start_frame_idx = config.start_frame_idx();
	m_step_frame = config.step_frame();

	// Camera-related
	m_downsample_scale = config.downsample_scale();
	m_cam2world = config.extrinsic()[0];

	// Visualize-related
	m_enable_vis = config.enable_vis();
	m_pcd_size = config.pcd_size();

	// Allocate buffer
	size_t num_pixel = config.raw_img_cols() * config.raw_img_rows();
	m_g_raw_color_img.create(num_pixel);
	m_g_raw_depth_img.create(num_pixel);

	cudaSafeCall(cudaMallocHost((void **)&m_raw_depth_img_buff, num_pixel * sizeof(unsigned short)));
	cudaSafeCall(cudaMallocHost((void **)&m_raw_color_img_buff, num_pixel * sizeof(uchar3)));
}

star::MeasureProcessorOffline::~MeasureProcessorOffline()
{
	m_g_raw_color_img.release();
	m_g_raw_depth_img.release();

	cudaSafeCall(cudaFreeHost(m_raw_depth_img_buff));
	cudaSafeCall(cudaFreeHost(m_raw_color_img_buff));
}

void star::MeasureProcessorOffline::ProcessFrame(
	const unsigned frame_idx,
	cudaStream_t stream)
{
	// 1. Load a rgb image && depth image
	const auto image_idx = size_t(frame_idx) * m_step_frame + m_start_frame_idx;
	m_fetcher->FetchRGBImage(0, image_idx, m_raw_color_img);
	m_fetcher->FetchDepthImage(0, image_idx, m_raw_depth_img);

	// CPU copy
	memcpy(m_raw_color_img_buff, m_raw_color_img.data,
		   sizeof(uchar3) * m_raw_color_img.total());
	memcpy(m_raw_depth_img_buff, m_raw_depth_img.data,
		   sizeof(unsigned short) * m_raw_depth_img.total());

	// Copy to GPU
	cudaSafeCall(cudaMemcpyAsync(
		m_g_raw_color_img.ptr(),
		m_raw_color_img_buff,
		m_raw_color_img.total() * sizeof(uchar3),
		cudaMemcpyHostToDevice,
		stream));
	cudaSafeCall(cudaMemcpyAsync(
		m_g_raw_depth_img.ptr(),
		m_raw_depth_img_buff,
		m_raw_depth_img.total() * sizeof(unsigned short),
		cudaMemcpyHostToDevice,
		stream));

	// 2. Initialize surfel map
	m_surfel_map_prev.swap(m_surfel_map);  // Replace cur & prev
	m_surfel_map_initializer->InitFromRGBDImage(
		GArrayView(m_g_raw_color_img),
		GArrayView(m_g_raw_depth_img),
		frame_idx,
		*m_surfel_map,
		stream);
	cudaSafeCall(cudaStreamSynchronize(stream));

	// 3. Visualize
	if (m_enable_vis)
		saveContext(frame_idx, stream);
}

void star::MeasureProcessorOffline::saveContext(
	const unsigned frame_idx,
	cudaStream_t stream)
{
	// Prepare
	auto &context = easy3d::Context::Instance();

	// Draw origin
	drawOrigin();

	// Draw point cloud
	context.addPointCloud("m_color", "", Eigen::Matrix4f::Identity(), m_pcd_size);
	visualize::SaveColoredPointCloud(
		m_surfel_map->VertexConfidReadOnly(),
		m_surfel_map->ColorTimeReadOnly(),
		context.at("m_color"));

	context.addPointCloud("m_normal", "", Eigen::Matrix4f::Identity(), m_pcd_size, "shadow");
	visualize::SavePointCloudWithNormal(
		m_surfel_map->VertexConfidReadOnly(),
		m_surfel_map->NormalRadiusReadOnly(),
		context.at("m_normal"));
	
	if (frame_idx != 0 && m_surfel_map_prev->NumValidSurfels() > 0) {
		context.addPointCloud("m_color_prev", "", Eigen::Matrix4f::Identity(), m_pcd_size);
		visualize::SaveColoredPointCloud(
			m_surfel_map_prev->VertexConfidReadOnly(),
			m_surfel_map_prev->ColorTimeReadOnly(),
			context.at("m_color_prev"));
	}

	// Save images
	context.addImage("measure-rgb");
	context.addImage("measure-depth");
	visualize::SaveNormalizeRGBDImage(m_surfel_map->Texture().rgbd, context.at("measure-rgb"), context.at("measure-depth"));
}

void star::MeasureProcessorOffline::drawOrigin()
{
	auto &context = easy3d::Context::Instance();

	// Draw Tsdf area
	Eigen::Matrix4f bb_center = Eigen::Matrix4f::Identity();
	float3 origin = make_float3(0.f, 0.f, 0.f);
	float3 box_size = make_float3(1.f, 1.f, 1.f);
	bb_center(0, 3) = origin.x + box_size.x / 2.f;
	bb_center(1, 3) = origin.y + box_size.y / 2.f;
	bb_center(2, 3) = origin.z + box_size.z / 2.f;
	context.addBoundingBox("bounding_box", "helper", bb_center, box_size.x, box_size.y, box_size.z);
	context.addCoord("origin", "helper", Eigen::Matrix4f::Identity(), 1.f);
}