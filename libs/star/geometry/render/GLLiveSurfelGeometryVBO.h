#pragma once
#include <star/geometry/render/glad/glad.h>
#include <star/geometry/surfel/SurfelGeometry.h>

namespace star
{

	/**
	 * \brief A struct to maintain all the
	 *        vertex buffer objects for a
	 *        instance of live surfel geometry,
	 *        alongside with reource for
	 *        access on cuda.
	 *        This struct can only be used
	 *        inside the render class.
	 */
	struct GLLiveSurfelGeometryVBO
	{
		// The vertex buffer objects correspond
		// to the member of LiveSurfelGeometry
		GLuint live_vertex_confid;
		GLuint live_normal_radius;
		GLuint color_time;

		// The cuda resource associated
		// with the surfel geomety vbos
		cudaGraphicsResource_t cuda_vbo_resources[3];

		// Methods can only be accessed by renderer
		void initialize();
		void release();

		void mapToCuda(LiveSurfelGeometry &geometry, cudaStream_t stream = 0);
		void mapToCuda(cudaStream_t stream = 0);

		// block all cuda calls in the given threads
		// for later OpenGL drawing pipelines
		void unmapFromCuda(cudaStream_t stream = 0);
	};

	// Use procedual for more clear ordering
	void initializeGLLiveSurfelGeometry(GLLiveSurfelGeometryVBO &surfel_vbo);
	void releaseGLLiveSurfelGeometry(GLLiveSurfelGeometryVBO &surfel_vbo);
}