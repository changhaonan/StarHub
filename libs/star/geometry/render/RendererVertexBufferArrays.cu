#include <star/common/logging.h>
#include <star/geometry/render/Renderer.h>
#include <star/geometry/render/GLSurfelGeometryVBO.h>
#include <star/geometry/render/GLSurfelGeometryVAO.h>

/* The vertex buffer objects mapping and selection and
 */
void star::Renderer::initModelVertexBufferObjects()
{
	initializeGLSurfelGeometry(m_model_surfel_geometry_vbos[0]);
	initializeGLSurfelGeometry(m_model_surfel_geometry_vbos[1]);
}

void star::Renderer::freeModelVertexBufferObjects()
{
	releaseGLSurfelGeometry(m_model_surfel_geometry_vbos[0]);
	releaseGLSurfelGeometry(m_model_surfel_geometry_vbos[1]);
}

void star::Renderer::MapModelSurfelGeometryToCuda(int idx, star::SurfelGeometry &geometry, cudaStream_t stream)
{
	idx = idx % 2;
	m_model_surfel_geometry_vbos[idx].mapToCuda(geometry, stream);
}

void star::Renderer::MapModelSurfelGeometryToCuda(int idx, cudaStream_t stream)
{
	idx = idx % 2;
	m_model_surfel_geometry_vbos[idx].mapToCuda(stream);
}

void star::Renderer::UnmapModelSurfelGeometryFromCuda(int idx, cudaStream_t stream)
{
	idx = idx % 2;
	m_model_surfel_geometry_vbos[idx].unmapFromCuda(stream);
}

/* The methods for vertex array objects
 */
void star::Renderer::initMapRenderVAO()
{
	// Each vao match one for vbos
	buildFusionMapVAO(m_model_surfel_geometry_vbos[0], m_fusion_map_vao[0]);
	buildFusionMapVAO(m_model_surfel_geometry_vbos[1], m_fusion_map_vao[1]);

	buildSolverMapVAO(m_model_surfel_geometry_vbos[0], m_solver_map_vao[0]);
	buildSolverMapVAO(m_model_surfel_geometry_vbos[1], m_solver_map_vao[1]);

	buildReferenceGeometryVAO(m_model_surfel_geometry_vbos[0], m_reference_geometry_vao[0]);
	buildReferenceGeometryVAO(m_model_surfel_geometry_vbos[1], m_reference_geometry_vao[1]);

	buildLiveGeometryVAO(m_model_surfel_geometry_vbos[0], m_live_geometry_vao[0]);
	buildLiveGeometryVAO(m_model_surfel_geometry_vbos[1], m_live_geometry_vao[1]);

	buildObservationMapVAO(m_model_surfel_geometry_vbos[0], m_observation_map_vao[0]);
	buildObservationMapVAO(m_model_surfel_geometry_vbos[1], m_observation_map_vao[1]);

	buildFilterMapVAO(m_data_surfel_geometry_vbos[0], m_filter_map_vao[0]);
	buildFilterMapVAO(m_data_surfel_geometry_vbos[1], m_filter_map_vao[1]);
}

void star::Renderer::initDataVertexBufferObjects()
{
	initializeGLSurfelGeometry(m_data_surfel_geometry_vbos[0]);
	initializeGLSurfelGeometry(m_data_surfel_geometry_vbos[1]);
}

void star::Renderer::freeDataVertexBufferObjects()
{
	releaseGLSurfelGeometry(m_data_surfel_geometry_vbos[0]);
	releaseGLSurfelGeometry(m_data_surfel_geometry_vbos[1]);
}

void star::Renderer::MapDataSurfelGeometryToCuda(int idx, star::SurfelGeometry &geometry, cudaStream_t stream)
{
	idx = idx % 2;
	m_data_surfel_geometry_vbos[idx].mapToCuda(geometry, stream);
}

void star::Renderer::MapDataSurfelGeometryToCuda(int idx, cudaStream_t stream)
{
	idx = idx % 2;
	m_data_surfel_geometry_vbos[idx].mapToCuda(stream);
}

void star::Renderer::UnmapDataSurfelGeometryFromCuda(int idx, cudaStream_t stream)
{
	idx = idx % 2;
	m_data_surfel_geometry_vbos[idx].unmapFromCuda(stream);
}