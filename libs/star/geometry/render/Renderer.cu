#include <star/geometry/render/Renderer.h>

star::Renderer::Renderer(
	const unsigned num_cam,
	const unsigned renderer_image_width,
	const unsigned renderer_image_height,
	const float4 &renderer_intrinsic,
	const float max_rendering_depth)
{
	if (!glfwInit())
	{
		LOG(FATAL) << "The graphic pipeline is not correctly initialized";
	}

	m_num_cam = num_cam;
	for (auto cam_idx = 0; cam_idx < m_num_cam; ++cam_idx)
	{
		m_downsample_image_width[cam_idx] = renderer_image_width;
		m_downsample_image_height[cam_idx] = renderer_image_height;
		m_fusion_map_width[cam_idx] = m_downsample_image_width[cam_idx] * d_fusion_map_scale;
		m_fusion_map_height[cam_idx] = m_downsample_image_height[cam_idx] * d_fusion_map_scale;
		m_renderer_intrinsic[cam_idx] = renderer_intrinsic;
		m_width_height_maxdepth[cam_idx] = make_float4(m_downsample_image_width[cam_idx], m_downsample_image_height[cam_idx], max_rendering_depth, 0.0f);
	}

	// A series of sub-init functions
	initGLFW();
	initClearValues();
	initModelVertexBufferObjects(); // The vertex buffer objects
	initDataVertexBufferObjects();	// Live VBO
	initMapRenderVAO();				// The vao, must after vbos
	initFrameRenderBuffers();
	initShaders();
}

star::Renderer::~Renderer()
{
	// A series of sub-free functions
	freeModelVertexBufferObjects();
	freeDataVertexBufferObjects();
	freeFrameRenderBuffers();
}

/* GLFW window related functions
 */
void star::Renderer::initGLFW()
{
	// The primary monitor
	mGLFWmonitor = glfwGetPrimaryMonitor();

	// The opengl context
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

	// Defualt framebuffer properties
	glfwWindowHint(GLFW_VISIBLE, GL_FALSE);
	glfwWindowHint(GLFW_SAMPLES, 1);
	glfwWindowHint(GLFW_STEREO, GL_FALSE);
	glfwWindowHint(GLFW_DOUBLEBUFFER, GL_TRUE);

	// Switch to second montior
	int monitor_count = 0;
	GLFWmonitor **monitors = glfwGetMonitors(&monitor_count);
	if (monitor_count > 1)
	{
		mGLFWmonitor = monitors[1];
	}

	// Setup of the window
	mGLFWwindow = glfwCreateWindow(1920, 720, "SurfelWarp", NULL, NULL);
	if (mGLFWwindow == NULL)
	{
		LOG(FATAL) << "The GLFW window is not correctly created";
	}

	// Make newly created context current
	glfwMakeContextCurrent(mGLFWwindow);

	// Init glad
	if (!gladLoadGL())
	{
		LOG(FATAL) << "Glad is not correctly initialized";
	}

	// Enable depth test, disable face culling
	glEnable(GL_DEPTH_TEST);
	glDisable(GL_CULL_FACE);
	glEnable(GL_PROGRAM_POINT_SIZE);
	glEnable(GL_POINT_SPRITE);
}

/* Initialize the value to clear the rendered images
 */
void star::Renderer::initClearValues()
{
	m_clear_values.initialize();
}

/* The method to initialize frame/render buffer object
 */
void star::Renderer::initFrameRenderBuffers()
{
	for (auto cam_idx = 0; cam_idx < m_num_cam; ++cam_idx)
	{
		m_fusion_map_buffers[cam_idx].initialize(m_fusion_map_width[cam_idx], m_fusion_map_height[cam_idx]);
		m_solver_map_buffers[cam_idx].initialize(m_downsample_image_width[cam_idx], m_downsample_image_height[cam_idx]);
		m_visualization_draw_buffers[cam_idx].initialize(m_downsample_image_width[cam_idx], m_downsample_image_height[cam_idx]);
		// observation
		m_observation_map_buffers[cam_idx].initialize(m_downsample_image_width[cam_idx], m_downsample_image_height[cam_idx]);
		m_filter_map_buffers[cam_idx].initialize(m_fusion_map_width[cam_idx], m_fusion_map_height[cam_idx]);
	}
}

void star::Renderer::freeFrameRenderBuffers()
{
	for (auto cam_idx = 0; cam_idx < m_num_cam; ++cam_idx)
	{
		m_fusion_map_buffers[cam_idx].release();
		m_solver_map_buffers[cam_idx].release();
		m_visualization_draw_buffers[cam_idx].release();
		// observation
		m_observation_map_buffers[cam_idx].release();
		m_filter_map_buffers[cam_idx].release();
	}
}

/* The access of fusion maps
 */
void star::Renderer::MapFusionMapsToCuda(FusionMaps &maps, cudaStream_t stream)
{
	for (auto cam_idx = 0; cam_idx < m_num_cam; ++cam_idx)
	{
		m_fusion_map_buffers[cam_idx].mapToCuda(
			maps.warp_vertex_confid_map[cam_idx],
			maps.warp_normal_radius_map[cam_idx],
			maps.index_map[cam_idx],
			maps.color_time_map[cam_idx],
			stream);
	}
}

void star::Renderer::MapFusionMapsToCuda(cudaStream_t stream)
{
	for (auto cam_idx = 0; cam_idx < m_num_cam; ++cam_idx)
	{
		m_fusion_map_buffers[cam_idx].mapToCuda(stream);
	}
}

void star::Renderer::UnmapFusionMapsFromCuda(cudaStream_t stream)
{
	for (auto cam_idx = 0; cam_idx < m_num_cam; ++cam_idx)
	{
		m_fusion_map_buffers[cam_idx].unmapFromCuda(stream);
	}
}

/* The texture access of solver maps
 */
void star::Renderer::MapSolverMapsToCuda(star::SolverMaps &maps, cudaStream_t stream)
{
	for (auto cam_idx = 0; cam_idx < m_num_cam; ++cam_idx)
	{
		m_solver_map_buffers[cam_idx].mapToCuda(
			maps.reference_vertex_map[cam_idx],
			maps.reference_normal_map[cam_idx],
			maps.index_map[cam_idx],
			maps.normalized_rgbd_map[cam_idx],
			stream);
	}
}

void star::Renderer::MapSolverMapsToCuda(cudaStream_t stream)
{
	for (auto cam_idx = 0; cam_idx < m_num_cam; ++cam_idx)
	{
		m_solver_map_buffers[cam_idx].mapToCuda(stream);
	}
}

void star::Renderer::UnmapSolverMapsFromCuda(cudaStream_t stream)
{
	for (auto cam_idx = 0; cam_idx < m_num_cam; ++cam_idx)
	{
		m_solver_map_buffers[cam_idx].unmapFromCuda(stream);
	}
}

void star::Renderer::MapObservationMapsToCuda(ObservationMaps &maps, cudaStream_t stream)
{
	for (auto cam_idx = 0; cam_idx < m_num_cam; ++cam_idx)
	{
		m_observation_map_buffers[cam_idx].mapToCuda(
			maps.rgbd_map[cam_idx],
			maps.index_map[cam_idx],
			stream);
	}
}

void star::Renderer::MapObservationMapsToCuda(cudaStream_t stream)
{
	for (auto cam_idx = 0; cam_idx < m_num_cam; ++cam_idx)
	{
		m_observation_map_buffers[cam_idx].mapToCuda(stream);
	}
}

void star::Renderer::UnmapObservationMapsFromCuda(cudaStream_t stream)
{
	for (auto cam_idx = 0; cam_idx < m_num_cam; ++cam_idx)
	{
		m_observation_map_buffers[cam_idx].unmapFromCuda(stream);
	}
}

void star::Renderer::MapFilterMapsToCuda(FilterMaps *maps, cudaStream_t stream)
{
	for (auto cam_idx = 0; cam_idx < m_num_cam; ++cam_idx)
	{
		m_filter_map_buffers[cam_idx].mapToCuda(
			maps[cam_idx].warp_vertex_confid_map,
			maps[cam_idx].warp_normal_radius_map,
			maps[cam_idx].index_map,
			maps[cam_idx].color_time_map,
			stream);
	}
}

void star::Renderer::MapFilterMapsToCuda(cudaStream_t stream)
{
	for (auto cam_idx = 0; cam_idx < m_num_cam; ++cam_idx)
	{
		m_filter_map_buffers[cam_idx].mapToCuda(stream);
	}
}

void star::Renderer::UnmapFilterMapsFromCuda(cudaStream_t stream)
{
	for (auto cam_idx = 0; cam_idx < m_num_cam; ++cam_idx)
	{
		m_filter_map_buffers[cam_idx].unmapFromCuda(stream);
	}
}