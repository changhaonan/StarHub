#pragma once
#include <star/geometry/render/GLLiveSurfelGeometryVBO.h>
#include <star/geometry/render/GLSurfelGeometryVBO.h>

namespace star
{

	// Init the vao for fusion map
	void buildFusionMapVAO(const GLSurfelGeometryVBO &geometryVBO, GLuint &fusion_map_vao);

	// Init the vao for warp solver
	void buildSolverMapVAO(const GLSurfelGeometryVBO &geometryVBO, GLuint &solver_map_vao);

	// Init the vao for reference geometry
	void buildReferenceGeometryVAO(const GLSurfelGeometryVBO &geometryVBO, GLuint &reference_geometry_vao);

	// Init the vao for live geometry
	void buildLiveGeometryVAO(const GLSurfelGeometryVBO &geometryVBO, GLuint &live_geometry_vbo);

	// Init the vao for observation ma
	void buildObservationMapVAO(const GLSurfelGeometryVBO &geometryVBO, GLuint &observation_map_vao);

	void buildFilterMapVAO(const GLSurfelGeometryVBO &geometryVBO, GLuint &filter_map_vao);
}
