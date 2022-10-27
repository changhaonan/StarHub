#pragma once
#include <star/geometry/render/glad/glad.h>
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

// Cuda headers
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cuda_gl_interop.h>

// STL headers
#include <tuple>
#include <vector>
#include <Eigen/Eigen>
#include <memory>

// The type decals
#include <star/common/constants.h>
#include <star/common/logging.h>
#include <star/common/common_types.h>
#include <star/geometry/constants.h>
#include <star/geometry/types/fusion_types.h>
#include <star/geometry/types/solver_types.h>
#include <star/geometry/types/measure_types.h>
#include <star/geometry/render/GLSurfelGeometryVBO.h>
#include <star/geometry/render/GLLiveSurfelGeometryVBO.h>
#include <star/geometry/render/GLRenderedMaps.h>
#include <star/geometry/render/GLClearValues.h>
#include <star/geometry/render/GLShaderProgram.h>

namespace star
{

	class Renderer
	{
	private:
		// These member should be obtained from the config parser
		unsigned m_num_cam;
		unsigned m_downsample_image_width[d_max_cam];
		unsigned m_downsample_image_height[d_max_cam];
		unsigned m_fusion_map_width[d_max_cam];
		unsigned m_fusion_map_height[d_max_cam];

		// The parameters that is accessed by drawing pipelines
		// The renderer intrinsic is a virtual intrinsic, thus, it is not the same as the physical
		float4 m_renderer_intrinsic[d_max_cam];
		float4 m_width_height_maxdepth[d_max_cam];

	public:
		// Accessed by pointer
		using Ptr = std::shared_ptr<Renderer>;
		explicit Renderer(
			const unsigned num_cam,
			const unsigned renderer_image_width,
			const unsigned renderer_image_height,
			const float4 &renderer_intrinsic,
			const float max_rendering_depth);
		~Renderer();
		STAR_NO_COPY_ASSIGN_MOVE(Renderer);

		/* GLFW windows related variables and functions
		 */
	private:
		GLFWmonitor *mGLFWmonitor = nullptr;
		GLFWwindow *mGLFWwindow = nullptr;
		void initGLFW();

		/* The buffer and method to clear the image
		 */
	private:
		GLClearValues m_clear_values;
		void initClearValues();

		/* The vertex buffer objects for surfel geometry
		 * Note that a double-buffer scheme is used here
		 */
	private:
		GLSurfelGeometryVBO m_model_surfel_geometry_vbos[2];
		void initModelVertexBufferObjects();
		void freeModelVertexBufferObjects();

	public:
		void MapModelSurfelGeometryToCuda(int idx, SurfelGeometry &geometry, cudaStream_t stream = 0);
		void MapModelSurfelGeometryToCuda(int idx, cudaStream_t stream = 0);
		void UnmapModelSurfelGeometryFromCuda(int idx, cudaStream_t stream = 0);

		/* The buffer for rendered maps
		 */
	private:
		// The frame/render buffer required for online processing
		GLFusionMapsFrameRenderBufferObjects m_fusion_map_buffers[d_max_cam];
		GLSolverMapsFrameRenderBufferObjects m_solver_map_buffers[d_max_cam];

		// The frame/render buffer for offline visualization
		GLOfflineVisualizationFrameRenderBufferObjects m_visualization_draw_buffers[d_max_cam];
		void initFrameRenderBuffers();
		void freeFrameRenderBuffers();

		/* The vao for rendering, must be init after
		 * the initialization of vbos
		 */
	private:
		// The vao for processing, correspond to double buffer scheme
		GLuint m_fusion_map_vao[2];
		GLuint m_solver_map_vao[2];

		// The vao for offline visualization of reference and live geometry
		GLuint m_reference_geometry_vao[2];
		GLuint m_live_geometry_vao[2];
		void initMapRenderVAO();

		/* The shader program to render the maps for
		 * solver and geometry updater
		 */
	private:
		GLShaderProgram m_fusion_map_shader;
		GLShaderProgram m_solver_map_shader; // This shader will draw recent observation
		void initProcessingShaders();

		// the collect of shaders for visualization
		struct
		{
			GLShaderProgram normal_map;
			GLShaderProgram phong_map;
			GLShaderProgram albedo_map;
		} m_visualization_shaders;
		void initVisualizationShaders();
		void initShaders();

		// The workforce method for solver maps drawing
		void drawSolverMaps(unsigned num_vertex, int vao_idx, int cam_idx, int current_time, const Matrix4f &world2camera, bool with_recent_observation);

		// The workforce method for offline visualization
		void drawVisualizationMap(
			GLShaderProgram &shader,
			GLuint geometry_vao, int cam_idx,
			unsigned num_vertex, int current_time,
			const Matrix4f &world2camera,
			bool with_recent_observation);

	public:
		void DrawFusionMaps(unsigned num_vertex, int vao_idx, int cam_idx, const Matrix4f &world2camera);
		void DrawSolverMapsConfidentObservation(unsigned num_vertex, int vao_idx, int cam_idx, int current_time, const Matrix4f &world2camera);
		void DrawSolverMapsWithRecentObservation(unsigned num_vertex, int vao_idx, int cam_idx, int current_time, const Matrix4f &world2camera);

		// The offline visualization methods
		void SaveLiveNormalMap(unsigned num_vertex, int vao_idx, int cam_idx, int current_time, const Matrix4f &world2camera, const std::string &path, bool with_recent = true);
		void SaveLiveAlbedoMap(unsigned num_vertex, int vao_idx, int cam_idx, int current_time, const Matrix4f &world2camera, const std::string &path, bool with_recent = true);
		void SaveLivePhongMap(unsigned num_vertex, int vao_idx, int cam_idx, int current_time, const Matrix4f &world2camera, const std::string &path, bool with_recent = true);
		void SaveReferenceNormalMap(unsigned num_vertex, int vao_idx, int cam_idx, int current_time, const Matrix4f &world2camera, const std::string &path, bool with_recent = true);
		void SaveReferenceAlbedoMap(unsigned num_vertex, int vao_idx, int cam_idx, int current_time, const Matrix4f &world2camera, const std::string &path, bool with_recent = true);
		void SaveReferencePhongMap(unsigned num_vertex, int vao_idx, int cam_idx, int current_time, const Matrix4f &world2camera, const std::string &path, bool with_recent = true);

		/* The access of fusion map
		 */
	public:
		void MapFusionMapsToCuda(FusionMaps &maps, cudaStream_t stream = 0);
		void MapFusionMapsToCuda(cudaStream_t stream = 0); // Re-connect
		void UnmapFusionMapsFromCuda(cudaStream_t stream = 0);

		/* The access of solver maps
		 */
	public:
		void MapSolverMapsToCuda(SolverMaps &maps, cudaStream_t stream = 0);
		void MapSolverMapsToCuda(cudaStream_t stream = 0);
		void UnmapSolverMapsFromCuda(cudaStream_t stream = 0);

	public:
		/* The access of observation map
		 */
	private:
		// The vao for filtering and fusion on observation side
		GLuint m_filter_map_vao[2];
		GLuint m_observation_map_vao[2];
		GLObservationMapsFrameRenderBufferObjects m_observation_map_buffers[d_max_cam];
		GLFusionMapsFrameRenderBufferObjects m_filter_map_buffers[d_max_cam];

	public:
		// Observation is measure but from geometry
		void MapObservationMapsToCuda(ObservationMaps &maps, cudaStream_t stream = 0);
		void MapObservationMapsToCuda(cudaStream_t stream = 0);
		void UnmapObservationMapsFromCuda(cudaStream_t stream = 0);

		/*
		 * Currently, filter map is fusion map;
		 * Could be changed in the future
		 */
	public:
		struct FilterMaps
		{
			cudaTextureObject_t warp_vertex_confid_map;
			cudaTextureObject_t warp_normal_radius_map;
			cudaTextureObject_t index_map;
			cudaTextureObject_t color_time_map;
		};
		void MapFilterMapsToCuda(FilterMaps *maps, cudaStream_t stream = 0);
		void MapFilterMapsToCuda(cudaStream_t stream = 0);
		void UnmapFilterMapsFromCuda(cudaStream_t stream = 0);

	private:
		// LiveGeometry
		GLSurfelGeometryVBO m_data_surfel_geometry_vbos[2];
		void initDataVertexBufferObjects();
		void freeDataVertexBufferObjects();

	public:
		void MapDataSurfelGeometryToCuda(int idx, SurfelGeometry &geometry, cudaStream_t stream = 0);
		void MapDataSurfelGeometryToCuda(int idx, cudaStream_t stream = 0);
		void UnmapDataSurfelGeometryFromCuda(int idx, cudaStream_t stream = 0);

	private:
		GLShaderProgram m_observation_map_shader;

	public:
		//  Drawing function
		void DrawObservationMaps(
			unsigned num_vertex,
			int vao_idx, int cam_idx, int current_time,
			const Matrix4f &world2camera,
			bool with_recent_observation);
		void DrawFilterMaps(unsigned num_vertex, int vao_idx, int cam_idx, const Matrix4f &world2camera);
	};
}