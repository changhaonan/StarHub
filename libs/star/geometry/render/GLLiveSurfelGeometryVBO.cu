#include <star/common/common_types.h>
#include <star/geometry/constants.h>
#include <star/geometry/render/GLLiveSurfelGeometryVBO.h>

// Register buffer to cuda
#include <cuda_gl_interop.h>

void star::GLLiveSurfelGeometryVBO::initialize()
{
	initializeGLLiveSurfelGeometry(*this);
}

void star::GLLiveSurfelGeometryVBO::release()
{
	releaseGLLiveSurfelGeometry(*this);
}

void star::GLLiveSurfelGeometryVBO::mapToCuda(
	star::LiveSurfelGeometry &geometry,
	cudaStream_t stream)
{
	// First map the resource
	cudaSafeCall(cudaGraphicsMapResources(3, cuda_vbo_resources, stream));

	// Get the buffer
	void *dptr;
	size_t buffer_size;
	const size_t num_valid_surfels = geometry.NumValidSurfels();

	// live vertex confidence
	cudaSafeCall(cudaGraphicsResourceGetMappedPointer(&dptr, &buffer_size, cuda_vbo_resources[0]));
	geometry.m_live_vertex_confid = GSliceBufferArray<float4>((float4 *)dptr, buffer_size / sizeof(float4), num_valid_surfels);

	// live normal-radius
	cudaSafeCall(cudaGraphicsResourceGetMappedPointer(&dptr, &buffer_size, cuda_vbo_resources[1]));
	geometry.m_live_normal_radius = GSliceBufferArray<float4>((float4 *)dptr, buffer_size / sizeof(float4), num_valid_surfels);

	// color time
	cudaSafeCall(cudaGraphicsResourceGetMappedPointer(&dptr, &buffer_size, cuda_vbo_resources[2]));
	geometry.m_color_time = GSliceBufferArray<float4>((float4 *)dptr, buffer_size / sizeof(float4), num_valid_surfels);
}

void star::GLLiveSurfelGeometryVBO::mapToCuda(cudaStream_t stream)
{
	cudaSafeCall(cudaGraphicsMapResources(3, cuda_vbo_resources, stream));
}

void star::GLLiveSurfelGeometryVBO::unmapFromCuda(cudaStream_t stream)
{
	cudaSafeCall(cudaGraphicsUnmapResources(3, cuda_vbo_resources, stream));
}

void star::initializeGLLiveSurfelGeometry(star::GLLiveSurfelGeometryVBO &vbo)
{
	glGenBuffers(1, &(vbo.live_vertex_confid));
	glGenBuffers(1, &(vbo.live_normal_radius));
	glGenBuffers(1, &(vbo.color_time));

	// Allocate buffers
	glBindBuffer(GL_ARRAY_BUFFER, vbo.live_vertex_confid);
	glBufferData(GL_ARRAY_BUFFER, 4 * sizeof(float) * d_max_num_surfels, NULL, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, vbo.live_normal_radius);
	glBufferData(GL_ARRAY_BUFFER, 4 * sizeof(float) * d_max_num_surfels, NULL, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, vbo.color_time);
	glBufferData(GL_ARRAY_BUFFER, 4 * sizeof(float) * d_max_num_surfels, NULL, GL_DYNAMIC_DRAW);

	// Register buffer to cuda
	cudaSafeCall(cudaGraphicsGLRegisterBuffer(&(vbo.cuda_vbo_resources[0]), vbo.live_vertex_confid, cudaGraphicsRegisterFlagsNone));
	cudaSafeCall(cudaGraphicsGLRegisterBuffer(&(vbo.cuda_vbo_resources[1]), vbo.live_normal_radius, cudaGraphicsRegisterFlagsNone));
	cudaSafeCall(cudaGraphicsGLRegisterBuffer(&(vbo.cuda_vbo_resources[2]), vbo.color_time, cudaGraphicsRegisterFlagsNone));
	cudaSafeCall(cudaGetLastError());
}

void star::releaseGLLiveSurfelGeometry(star::GLLiveSurfelGeometryVBO &vbo)
{
	// First un-register the resource
	cudaSafeCall(cudaGraphicsUnregisterResource(vbo.cuda_vbo_resources[0]));
	cudaSafeCall(cudaGraphicsUnregisterResource(vbo.cuda_vbo_resources[1]));
	cudaSafeCall(cudaGraphicsUnregisterResource(vbo.cuda_vbo_resources[2]));

	// Then we can delete the vbos
	glDeleteBuffers(1, &(vbo.live_vertex_confid));
	glDeleteBuffers(1, &(vbo.live_normal_radius));
	glDeleteBuffers(1, &(vbo.color_time));
}