#include <star/geometry/surfel/SurfelGeometry.h>
#include <device_launch_parameters.h>

namespace star::device
{
	__global__ void applySE3DebugKernel(
		const mat34 se3,
		GArraySlice<float4> referece_vertex_confid,
		float4 *reference_normal_radius,
		float4 *live_vertex_confid,
		float4 *live_normal_radius)
	{
		const auto idx = threadIdx.x + blockDim.x * blockIdx.x;
		if (idx < referece_vertex_confid.Size())
		{
			float4 ref_v4 = referece_vertex_confid[idx];
			float3 transformed_ref_v3 = se3.rot * ref_v4 + se3.trans;
			referece_vertex_confid[idx] = make_float4(transformed_ref_v3.x, transformed_ref_v3.y, transformed_ref_v3.z, ref_v4.w);

			float4 ref_n4 = reference_normal_radius[idx];
			float3 transformed_ref_n3 = se3.rot * ref_n4;
			reference_normal_radius[idx] = make_float4(transformed_ref_n3.x, transformed_ref_n3.y, transformed_ref_n3.z, ref_n4.w);

			float4 live_v4 = live_vertex_confid[idx];
			float3 transformed_live_v3 = se3.rot * live_v4 + se3.trans;
			live_vertex_confid[idx] = make_float4(transformed_live_v3.x, transformed_live_v3.y, transformed_live_v3.z, live_v4.w);

			float4 live_n4 = live_normal_radius[idx];
			float3 transformed_live_n3 = se3.rot * live_n4;
			live_normal_radius[idx] = make_float4(transformed_live_n3.x, transformed_live_n3.y, transformed_live_n3.z, live_n4.w);
		}
	}
}

void star::SurfelGeometry::AddSE3ToVertexAndNormalDebug(const mat34 &se3)
{
	dim3 blk(128);
	dim3 grid(divUp(NumValidSurfels(), blk.x));
	device::applySE3DebugKernel<<<grid, blk>>>(
		se3,
		m_reference_vertex_confid.Slice(),
		m_reference_normal_radius.Ptr(),
		m_live_vertex_confid.Ptr(),
		m_live_normal_radius.Ptr());

	cudaSafeCall(cudaDeviceSynchronize());
	cudaSafeCall(cudaGetLastError());
}

star::SurfelGeometry::SurfelGeometry() : m_num_valid_surfels(0)
{
	// KNN structure
	m_surfel_knn.AllocateBuffer(d_max_num_surfels);
	m_surfel_knn_spatial_weight.AllocateBuffer(d_max_num_surfels);
	m_surfel_knn_connect_weight.AllocateBuffer(d_max_num_surfels);

	// Resize all arrays to zero
	m_reference_vertex_confid.ResizeArrayOrException(0);
	m_reference_normal_radius.ResizeArrayOrException(0);
	m_live_vertex_confid.ResizeArrayOrException(0);
	m_live_normal_radius.ResizeArrayOrException(0);
	m_color_time.ResizeArrayOrException(0);

	// Optional
	m_semantic_prob.AllocateBuffer(d_max_num_surfels);
	m_semantic_prob.ResizeArrayOrException(0);
	cudaSafeCall(cudaMemset((void *)m_semantic_prob.Ptr(), 0, m_semantic_prob.BufferByteSize()));
}

star::SurfelGeometry::~SurfelGeometry()
{
	m_surfel_knn.ReleaseBuffer();
	m_surfel_knn_spatial_weight.ReleaseBuffer();
	m_surfel_knn_connect_weight.ReleaseBuffer();

	// Optional
	m_semantic_prob.ReleaseBuffer();
}

void star::SurfelGeometry::ResizeValidSurfelArrays(size_t size)
{
	// Resize non-owned geometry
	m_reference_vertex_confid.ResizeArrayOrException(size);
	m_reference_normal_radius.ResizeArrayOrException(size);
	m_live_vertex_confid.ResizeArrayOrException(size);
	m_live_normal_radius.ResizeArrayOrException(size);
	m_color_time.ResizeArrayOrException(size);
	// Optional
	m_semantic_prob.ResizeArrayOrException(size);

	m_surfel_knn.ResizeArrayOrException(size);
	m_surfel_knn_spatial_weight.ResizeArrayOrException(size);
	m_surfel_knn_connect_weight.ResizeArrayOrException(size);
	// Everything is ok
	m_num_valid_surfels = size;
}

/* The debug methods
 */
star::SurfelGeometry::GeometryAttributes star::SurfelGeometry::Geometry()
{
	GeometryAttributes geometry_attributes;
	geometry_attributes.reference_vertex_confid = m_reference_vertex_confid.Slice();
	geometry_attributes.reference_normal_radius = m_reference_normal_radius.Slice();
	geometry_attributes.live_vertex_confid = m_live_vertex_confid.Slice();
	geometry_attributes.live_normal_radius = m_live_normal_radius.Slice();
	geometry_attributes.color_time = m_color_time.Slice();
	geometry_attributes.semantic_prob = m_semantic_prob.Slice();
	return geometry_attributes;
}

/* The static methods
 */
void star::SurfelGeometry::ReAnchor(
	SurfelGeometry::ConstPtr src_geometry,
	SurfelGeometry::Ptr tar_geometry,
	cudaStream_t stream)
{
	// Copy owned data
	cudaSafeCall(
		cudaMemcpyAsync(
			tar_geometry->SurfelKNN(),
			src_geometry->SurfelKNNReadOnly(),
			src_geometry->SurfelKNNReadOnly().ByteSize(),
			cudaMemcpyDeviceToDevice, stream));
	cudaSafeCall(
		cudaMemcpyAsync(
			tar_geometry->SurfelKNNSpatialWeight(),
			src_geometry->SurfelKNNSpatialWeightReadOnly(),
			src_geometry->SurfelKNNSpatialWeightReadOnly().ByteSize(),
			cudaMemcpyDeviceToDevice, stream));
	cudaSafeCall(
		cudaMemcpyAsync(
			tar_geometry->SurfelKNNConnectWeight(),
			src_geometry->SurfelKNNConnectWeightReadOnly(),
			src_geometry->SurfelKNNConnectWeightReadOnly().ByteSize(),
			cudaMemcpyDeviceToDevice, stream));
	// Copy geometry data
	cudaSafeCall(
		cudaMemcpyAsync(
			tar_geometry->ReferenceVertexConfidence(),
			src_geometry->LiveVertexConfidenceReadOnly(),
			src_geometry->LiveVertexConfidenceReadOnly().ByteSize(),
			cudaMemcpyDeviceToDevice, stream));
	cudaSafeCall(
		cudaMemcpyAsync(
			tar_geometry->ReferenceNormalRadius(),
			src_geometry->LiveNormalRadiusReadOnly(),
			src_geometry->LiveNormalRadiusReadOnly().ByteSize(),
			cudaMemcpyDeviceToDevice, stream));
	cudaSafeCall(
		cudaMemcpyAsync(
			tar_geometry->ColorTime(),
			src_geometry->ColorTimeReadOnly(),
			src_geometry->ColorTimeReadOnly().ByteSize(),
			cudaMemcpyDeviceToDevice, stream));
	// Optional
	cudaSafeCall(
		cudaMemcpyAsync(
			tar_geometry->SemanticProb(),
			src_geometry->SemanticProbReadOnly(),
			src_geometry->SemanticProbReadOnly().ByteSize(),
			cudaMemcpyDeviceToDevice, stream));

	// Sync & Resize
	cudaSafeCall(cudaStreamSynchronize(stream));
	tar_geometry->ResizeValidSurfelArrays(src_geometry->NumValidSurfels());
}

/*
 * Live Geometry
 */
star::LiveSurfelGeometry::LiveSurfelGeometry() : m_num_valid_surfels(0)
{
	// Resize all arrays to zero
	m_live_vertex_confid.ResizeArrayOrException(0);
	m_live_normal_radius.ResizeArrayOrException(0);
	m_color_time.ResizeArrayOrException(0);
}

star::LiveSurfelGeometry::LiveSurfelGeometry(const SurfelGeometry &surfel_geometry)
	: m_num_valid_surfels(surfel_geometry.m_num_valid_surfels)
{
	m_live_vertex_confid = surfel_geometry.m_live_vertex_confid;
	m_live_normal_radius = surfel_geometry.m_live_normal_radius;
	m_color_time = surfel_geometry.m_color_time;
}

star::LiveSurfelGeometry::~LiveSurfelGeometry()
{ // Do nothing here
}

void star::LiveSurfelGeometry::ResizeValidSurfelArrays(size_t size)
{
	// Resize non-owned geometry
	m_live_vertex_confid.ResizeArrayOrException(size);
	m_live_normal_radius.ResizeArrayOrException(size);
	m_color_time.ResizeArrayOrException(size);

	// Everything is ok
	m_num_valid_surfels = size;
}

star::SurfelGeometry::Geometry4Solver star::SurfelGeometry::GenerateGeometry4Solver() const
{
	Geometry4Solver geometry4solver;
	geometry4solver.surfel_knn = m_surfel_knn.View();
	geometry4solver.surfel_knn_spatial_weight = m_surfel_knn_spatial_weight.View();
	geometry4solver.surfel_knn_connect_weight = m_surfel_knn_connect_weight.View();
	geometry4solver.num_vertex = m_num_valid_surfels;
	return geometry4solver;
}

star::Geometry4Skinner star::SurfelGeometry::GenerateGeometry4Skinner()
{
	Geometry4Skinner geometry4skinner;
	geometry4skinner.reference_vertex_confid = m_reference_vertex_confid.View();
	geometry4skinner.reference_normal_radius = m_reference_normal_radius.View();
	geometry4skinner.live_vertex_confid = m_live_vertex_confid.View();
	geometry4skinner.live_normal_radius = m_live_normal_radius.View();
	geometry4skinner.surfel_semantic_prob = m_semantic_prob.View();
	geometry4skinner.surfel_knn = m_surfel_knn.Slice();
	geometry4skinner.surfel_knn_spatial_weight = m_surfel_knn_spatial_weight.Slice();
	geometry4skinner.surfel_knn_connect_weight = m_surfel_knn_connect_weight.Slice();
	return geometry4skinner;
}

star::SurfelGeometry::Geometry4Fusion star::SurfelGeometry::GenerateGeometry4Fusion(const bool use_ref) {
	Geometry4Fusion geometry4fusion;
	if (use_ref) {
		geometry4fusion.vertex_confid = m_reference_vertex_confid.Slice();
		geometry4fusion.normal_radius = m_reference_normal_radius.Slice();
	}
	else {
		geometry4fusion.vertex_confid = m_live_vertex_confid.Slice();
		geometry4fusion.normal_radius = m_live_normal_radius.Slice();
	}
	geometry4fusion.color_time = m_color_time.Slice();
	geometry4fusion.num_valid_surfel = NumValidSurfels();
	return geometry4fusion;
}

star::SurfelGeometry::Geometry4SemanticFusion star::SurfelGeometry::GenerateGeometry4SemanticFusion() {
	Geometry4SemanticFusion geometry4semantic_fusion;
	geometry4semantic_fusion.semantic_prob = m_semantic_prob.Slice();
	geometry4semantic_fusion.num_valid_surfel = NumValidSurfels();
	return geometry4semantic_fusion;
}

/* The debug methods
 */
star::LiveSurfelGeometry::GeometryAttributes star::LiveSurfelGeometry::Geometry() const
{
	GeometryAttributes geometry_attributes;
	geometry_attributes.live_vertex_confid = m_live_vertex_confid.Slice();
	geometry_attributes.live_normal_radius = m_live_normal_radius.Slice();
	geometry_attributes.color_time = m_color_time.Slice();
	return geometry_attributes;
}

star::SurfelGeometrySC::SurfelGeometrySC() : SurfelGeometry()
{
	// Allocate buffer
	m_reference_vertex_confid_buffer.AllocateBuffer(d_max_num_surfels);
	m_reference_normal_radius_buffer.AllocateBuffer(d_max_num_surfels);
	m_live_vertex_confid_buffer.AllocateBuffer(d_max_num_surfels);
	m_live_normal_radius_buffer.AllocateBuffer(d_max_num_surfels);
	m_color_time_buffer.AllocateBuffer(d_max_num_surfels);

	// Bind buffer
	m_reference_vertex_confid = GSliceBufferArray(m_reference_vertex_confid_buffer.Ptr(), m_reference_vertex_confid_buffer.BufferSize());
	m_reference_normal_radius = GSliceBufferArray(m_reference_normal_radius_buffer.Ptr(), m_reference_normal_radius_buffer.BufferSize());
	m_live_vertex_confid = GSliceBufferArray(m_live_vertex_confid_buffer.Ptr(), m_live_vertex_confid_buffer.BufferSize());
	m_live_normal_radius = GSliceBufferArray(m_live_normal_radius_buffer.Ptr(), m_live_normal_radius_buffer.BufferSize());
	m_color_time = GSliceBufferArray(m_color_time_buffer.Ptr(), m_color_time_buffer.BufferSize());
}

star::SurfelGeometrySC::~SurfelGeometrySC()
{
	m_reference_vertex_confid_buffer.ReleaseBuffer();
	m_reference_normal_radius_buffer.ReleaseBuffer();
	m_live_vertex_confid_buffer.ReleaseBuffer();
	m_live_normal_radius_buffer.ReleaseBuffer();
	m_color_time_buffer.ReleaseBuffer();
}