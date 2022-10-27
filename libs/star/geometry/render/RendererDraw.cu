#include <boost/filesystem.hpp>
#include <star/geometry/render/Renderer.h>
#include <star/geometry/constants.h>

/* Compile and initialize the shader program
 * used for later drawing methods
 */
void star::Renderer::initProcessingShaders()
{
	// Query the shader path
	boost::filesystem::path file_path(__FILE__);
	const boost::filesystem::path render_path = file_path.parent_path();
	const boost::filesystem::path shader_path = render_path / "shaders";

	// Compile the shader for fusion map
	const auto fusion_map_vert_path = shader_path / "fusion_map.vert";
	const auto fusion_map_frag_path = shader_path / "fusion_map.frag";
	m_fusion_map_shader.Compile(fusion_map_vert_path.string(), fusion_map_frag_path.string());

	// Compile the shader for solver maps
	const auto solver_map_frag_path = shader_path / "solver_map.frag";

	// Use dense solver maps or not
#if defined(USE_DENSE_SOLVER_MAPS)
	const auto solver_map_recent_vert_path = shader_path / "solver_map_sized.vert";
#else
	const auto solver_map_recent_vert_path = shader_path / "solver_map.vert";
#endif
	m_solver_map_shader.Compile(solver_map_recent_vert_path.string(), solver_map_frag_path.string());

	// Compile shader for observation maps
	const auto observation_map_vert_path = shader_path / "geometry.vert";
	const auto observation_map_frag_path = shader_path / "observation_map.frag";
	m_observation_map_shader.Compile(observation_map_vert_path.string(), observation_map_frag_path.string());
}

void star::Renderer::initVisualizationShaders()
{
	boost::filesystem::path file_path(__FILE__);
	const boost::filesystem::path render_path = file_path.parent_path();
	const boost::filesystem::path shader_path = render_path / "shaders";

	// The fragment shader for normal map, phong shading and albedo color
	const auto normal_map_frag_path = shader_path / "normal_as_color.frag";
	const auto phong_shading_path = shader_path / "phong_color.frag";
	const auto albedo_color_path = shader_path / "albedo_color.frag";

	// The vertex shader for referenc and live geometry
	const auto geometry_vert_path = shader_path / "geometry.vert";

	// Compile the shader without recent observation
	m_visualization_shaders.normal_map.Compile(geometry_vert_path.string(), normal_map_frag_path.string());
	m_visualization_shaders.phong_map.Compile(geometry_vert_path.string(), phong_shading_path.string());
	m_visualization_shaders.albedo_map.Compile(geometry_vert_path.string(), albedo_color_path.string());
}

void star::Renderer::initShaders()
{
	// The shader for warp solver, data fusion and visualization
	initProcessingShaders();
	initVisualizationShaders();
}

/* The drawing and debug function for fusion map
 */
void star::Renderer::DrawFusionMaps(
	unsigned num_vertex,
	int vao_idx, int cam_idx,
	const Matrix4f &world2camera)
{
	// Bind the shader
	m_fusion_map_shader.Bind();

	// The vao/vbo for the rendering
	vao_idx = vao_idx % 2;
	glBindVertexArray(m_fusion_map_vao[vao_idx]);

	// The framebuffer
	glBindFramebuffer(GL_FRAMEBUFFER, m_fusion_map_buffers[cam_idx].fusion_map_fbo);
	glViewport(0, 0, m_fusion_map_width[cam_idx], m_fusion_map_height[cam_idx]);

	// Clear the render buffer object
	glClearBufferfv(GL_COLOR, 0, m_clear_values.vertex_map_clear);
	glClearBufferfv(GL_COLOR, 1, m_clear_values.normal_map_clear);
	glClearBufferuiv(GL_COLOR, 2, &(m_clear_values.index_map_clear));
	glClearBufferfv(GL_COLOR, 3, m_clear_values.color_time_clear);
	glClearBufferfv(GL_DEPTH, 0, &(m_clear_values.z_buffer_clear));

	// Set the uniform values
	m_fusion_map_shader.SetUniformMatrix("world2camera", world2camera);
	m_fusion_map_shader.SetUniformVector("intrinsic", m_renderer_intrinsic[cam_idx]);
	m_fusion_map_shader.SetUniformVector("width_height_maxdepth", m_width_height_maxdepth[cam_idx]);

	// Draw it
	glDrawArrays(GL_POINTS, 0, num_vertex);

	// Cleanup-code
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	glBindVertexArray(0);
	m_fusion_map_shader.Unbind();
}

/* The method to draw solver maps
 */
void star::Renderer::drawSolverMaps(
	unsigned num_vertex,
	int vao_idx, int cam_idx,
	int current_time,
	const star::Matrix4f &world2camera,
	bool with_recent_observation)
{
	// Bind the shader
	m_solver_map_shader.Bind();

	// Normalize the vao index
	vao_idx %= 2;
	glBindVertexArray(m_solver_map_vao[vao_idx]);

	// The size is image rows/cols
	glBindFramebuffer(GL_FRAMEBUFFER, m_solver_map_buffers[cam_idx].solver_map_fbo);
	glViewport(0, 0, m_downsample_image_width[cam_idx], m_downsample_image_height[cam_idx]);

	// Clear the render buffer object
	glClearBufferfv(GL_COLOR, 0, m_clear_values.vertex_map_clear);
	glClearBufferfv(GL_COLOR, 1, m_clear_values.normal_map_clear);
	glClearBufferuiv(GL_COLOR, 2, &(m_clear_values.index_map_clear));
	glClearBufferfv(GL_COLOR, 3, m_clear_values.solver_rgba_clear);
	glClearBufferfv(GL_DEPTH, 0, &(m_clear_values.z_buffer_clear));

	// Set uniform values
	m_solver_map_shader.SetUniformMatrix("world2camera", world2camera);
	m_solver_map_shader.SetUniformVector("intrinsic", m_renderer_intrinsic[cam_idx]);

	// The current time of the solver maps
	float4 width_height_maxdepth_currtime = make_float4(
		m_width_height_maxdepth[cam_idx].x, m_width_height_maxdepth[cam_idx].y,
		m_width_height_maxdepth[cam_idx].z, current_time);
	m_solver_map_shader.SetUniformVector("width_height_maxdepth_currtime", width_height_maxdepth_currtime);

	// The time threshold depend on input
	float2 confid_time_threshold = make_float2(d_stable_surfel_confidence_threshold, d_rendering_recent_time_threshold);
	if (!with_recent_observation)
	{
		confid_time_threshold.y = -1.0f; // Do not pass any surfel due to recent observed
	}

	m_solver_map_shader.SetUniformVector("confid_time_threshold", confid_time_threshold);

	// Draw it
	glDrawArrays(GL_POINTS, 0, num_vertex);

	// Cleanup-code
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	glBindVertexArray(0);
	m_solver_map_shader.Unbind();
}

void star::Renderer::DrawSolverMapsConfidentObservation(
	unsigned num_vertex,
	int vao_idx, int cam_idx,
	int current_time, // Not used
	const star::Matrix4f &world2camera)
{
	drawSolverMaps(num_vertex, vao_idx, cam_idx, current_time, world2camera, false);
}

void star::Renderer::DrawSolverMapsWithRecentObservation(
	unsigned num_vertex,
	int vao_idx, int cam_idx,
	int current_time,
	const star::Matrix4f &world2camera)
{
	drawSolverMaps(num_vertex, vao_idx, cam_idx, current_time, world2camera, true);
}

/* The method for visualization map drawing
 */
void star::Renderer::drawVisualizationMap(
	GLShaderProgram &shader,
	GLuint geometry_vao, int cam_idx,
	unsigned num_vertex, int current_time,
	const Matrix4f &world2camera,
	bool with_recent_observation)
{
	// Bind the shader
	shader.Bind();

	// Use the provided vao
	glBindVertexArray(geometry_vao);

	// The size is image rows/cols
	glBindFramebuffer(GL_FRAMEBUFFER, m_visualization_draw_buffers[cam_idx].visualization_map_fbo);
	glViewport(0, 0, m_downsample_image_width[cam_idx], m_downsample_image_height[cam_idx]);

	// Clear the render buffer object
	glClearBufferfv(GL_COLOR, 0, m_clear_values.visualize_rgba_clear);
	glClearBufferfv(GL_DEPTH, 0, &(m_clear_values.z_buffer_clear));

	// Set uniform values
	shader.SetUniformMatrix("world2camera", world2camera);
	shader.SetUniformVector("intrinsic", m_renderer_intrinsic[cam_idx]);

	// The current time of the solver maps
	const float4 width_height_maxdepth_currtime = make_float4(
		m_width_height_maxdepth[cam_idx].x, m_width_height_maxdepth[cam_idx].y,
		m_width_height_maxdepth[cam_idx].z, current_time);
	shader.SetUniformVector("width_height_maxdepth_currtime", width_height_maxdepth_currtime);

	// The time threshold depend on input
	float2 confid_time_threshold = make_float2(d_stable_surfel_confidence_threshold, d_rendering_recent_time_threshold);
	if (!with_recent_observation)
	{
		confid_time_threshold.y = -1.0f; // Do not pass any surfel due to recent observed
	}

	// Hand in to shader
	shader.SetUniformVector("confid_time_threshold", confid_time_threshold);

	// Draw it
	glDrawArrays(GL_POINTS, 0, num_vertex);

	// Cleanup code
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	glBindVertexArray(0);
	shader.Unbind();
}

void star::Renderer::SaveLiveNormalMap(
	unsigned num_vertex,
	int vao_idx, int cam_idx,
	int current_time,
	const star::Matrix4f &world2camera,
	const std::string &path,
	bool with_recent)
{
	// Draw it
	const auto geometry_vao = m_live_geometry_vao[vao_idx];
	drawVisualizationMap(
		m_visualization_shaders.normal_map,
		geometry_vao, cam_idx,
		num_vertex, current_time,
		world2camera,
		with_recent);

	// Save it
	m_visualization_draw_buffers[cam_idx].save(path);
}

void star::Renderer::SaveLiveAlbedoMap(
	unsigned num_vertex,
	int vao_idx, int cam_idx,
	int current_time,
	const star::Matrix4f &world2camera,
	const std::string &path,
	bool with_recent)
{
	// Draw it
	const auto geometry_vao = m_live_geometry_vao[vao_idx];
	drawVisualizationMap(
		m_visualization_shaders.albedo_map,
		geometry_vao, cam_idx,
		num_vertex, current_time,
		world2camera,
		with_recent);

	// Save it
	m_visualization_draw_buffers[cam_idx].save(path);
}

void star::Renderer::SaveLivePhongMap(
	unsigned num_vertex,
	int vao_idx, int cam_idx,
	int current_time,
	const star::Matrix4f &world2camera,
	const std::string &path,
	bool with_recent)
{
	// Draw it
	const auto geometry_vao = m_live_geometry_vao[vao_idx];
	drawVisualizationMap(
		m_visualization_shaders.phong_map,
		geometry_vao, cam_idx,
		num_vertex, current_time,
		world2camera,
		with_recent);

	// Save it
	m_visualization_draw_buffers[cam_idx].save(path);
}

void star::Renderer::SaveReferenceNormalMap(
	unsigned num_vertex,
	int vao_idx, int cam_idx,
	int current_time,
	const Matrix4f &world2camera,
	const std::string &path,
	bool with_recent)
{
	const auto geometry_vao = m_reference_geometry_vao[vao_idx];
	drawVisualizationMap(
		m_visualization_shaders.normal_map,
		geometry_vao, cam_idx,
		num_vertex, current_time,
		world2camera,
		with_recent);

	// Save it
	m_visualization_draw_buffers[cam_idx].save(path);
}

void star::Renderer::SaveReferenceAlbedoMap(
	unsigned num_vertex,
	int vao_idx, int cam_idx,
	int current_time,
	const Matrix4f &world2camera,
	const std::string &path,
	bool with_recent)
{
	const auto geometry_vao = m_reference_geometry_vao[vao_idx];
	drawVisualizationMap(
		m_visualization_shaders.albedo_map,
		geometry_vao, cam_idx,
		num_vertex, current_time,
		world2camera,
		with_recent);

	// Save it
	m_visualization_draw_buffers[cam_idx].save(path);
}

void star::Renderer::SaveReferencePhongMap(
	unsigned num_vertex,
	int vao_idx, int cam_idx,
	int current_time,
	const Matrix4f &world2camera,
	const std::string &path,
	bool with_recent)
{
	const auto geometry_vao = m_reference_geometry_vao[vao_idx];
	drawVisualizationMap(
		m_visualization_shaders.phong_map,
		geometry_vao, cam_idx,
		num_vertex, current_time,
		world2camera,
		with_recent);

	// Save it
	m_visualization_draw_buffers[cam_idx].save(path);
}

/* The drawing and debug function for observation map
 */
void star::Renderer::DrawObservationMaps(
	unsigned num_vertex,
	int vao_idx, int cam_idx, int current_time,
	const Matrix4f &world2camera,
	bool with_recent_observation)
{
	// Bind the shader
	m_observation_map_shader.Bind();

	// The vao/vbo for the rendering
	vao_idx = vao_idx % 2;
	glBindVertexArray(m_observation_map_vao[vao_idx]);

	// The framebuffer
	glBindFramebuffer(GL_FRAMEBUFFER, m_observation_map_buffers[cam_idx].observation_map_fbo);
	glViewport(0, 0, m_downsample_image_width[cam_idx], m_downsample_image_height[cam_idx]);

	// Clear the render buffer object
	glClearBufferfv(GL_COLOR, 0, m_clear_values.rgbd_map_clear);
	glClearBufferuiv(GL_COLOR, 1, &(m_clear_values.index_map_clear));
	glClearBufferfv(GL_DEPTH, 0, &(m_clear_values.z_buffer_clear));

	// Set the uniform values
	m_observation_map_shader.SetUniformMatrix("world2camera", world2camera);
	m_observation_map_shader.SetUniformVector("intrinsic", m_renderer_intrinsic[cam_idx]);
	// The current time of the solver maps
	const float4 width_height_maxdepth_currtime = make_float4(
		m_width_height_maxdepth[cam_idx].x, m_width_height_maxdepth[cam_idx].y,
		m_width_height_maxdepth[cam_idx].z, current_time);
	m_observation_map_shader.SetUniformVector("width_height_maxdepth_currtime", width_height_maxdepth_currtime);

	// The time threshold depend on input
	float2 confid_time_threshold = make_float2(d_stable_surfel_confidence_threshold, d_rendering_recent_time_threshold);
	if (!with_recent_observation)
	{
		confid_time_threshold.y = -1.0f; // Do not pass any surfel due to recent observed
	}
	// Hand in to shader
	m_observation_map_shader.SetUniformVector("confid_time_threshold", confid_time_threshold);

	// Draw it
	glDrawArrays(GL_POINTS, 0, num_vertex);

	// Cleanup-code
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	glBindVertexArray(0);
	m_observation_map_shader.Unbind();
}

void star::Renderer::DrawFilterMaps(unsigned num_vertex, int vao_idx, int cam_idx, const Matrix4f &world2camera)
{
	// Bind the shader
	m_fusion_map_shader.Bind();

	// The vao/vbo for the rendering
	vao_idx = vao_idx % 2;
	glBindVertexArray(m_filter_map_vao[vao_idx]);

	// The framebuffer
	glBindFramebuffer(GL_FRAMEBUFFER, m_filter_map_buffers[cam_idx].fusion_map_fbo);
	glViewport(0, 0, m_fusion_map_width[cam_idx], m_fusion_map_height[cam_idx]);

	// Clear the render buffer object
	glClearBufferfv(GL_COLOR, 0, m_clear_values.vertex_map_clear);
	glClearBufferfv(GL_COLOR, 1, m_clear_values.normal_map_clear);
	glClearBufferuiv(GL_COLOR, 2, &(m_clear_values.index_map_clear));
	glClearBufferfv(GL_COLOR, 3, m_clear_values.color_time_clear);
	glClearBufferfv(GL_DEPTH, 0, &(m_clear_values.z_buffer_clear));

	// Set the uniform values
	m_fusion_map_shader.SetUniformMatrix("world2camera", world2camera);
	m_fusion_map_shader.SetUniformVector("intrinsic", m_renderer_intrinsic[cam_idx]);
	m_fusion_map_shader.SetUniformVector("width_height_maxdepth", m_width_height_maxdepth[cam_idx]);

	// Draw it
	glDrawArrays(GL_POINTS, 0, num_vertex);

	// Cleanup-code
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	glBindVertexArray(0);
	m_fusion_map_shader.Unbind();
}
