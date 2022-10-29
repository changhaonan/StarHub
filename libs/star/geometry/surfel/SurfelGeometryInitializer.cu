#include <star/geometry/surfel/SurfelGeometryInitializer.h>
#include <device_launch_parameters.h>

namespace star::device
{
	__global__ void initializerCollectDepthSurfelKernel(
		GArrayView<DepthSurfel> surfel_array,
		float4 *reference_vertex_confid,
		float4 *reference_normal_radius,
		float4 *live_vertex_confid,
		float4 *live_normal_radius,
		float4 *color_time)
	{
		const auto idx = threadIdx.x + blockDim.x * blockIdx.x;
		if (idx < surfel_array.Size())
		{
			const DepthSurfel &surfel = surfel_array[idx];
			reference_vertex_confid[idx] = live_vertex_confid[idx] = surfel.vertex_confid;
			reference_normal_radius[idx] = live_normal_radius[idx] = surfel.normal_radius;
			color_time[idx] = surfel.color_time;
		}
	}

	__global__ void initializerExpandDepthSurfelKernel(
		GArrayView<DepthSurfel> surfel_array,
		float4 *reference_vertex_confid,
		float4 *reference_normal_radius,
		float4 *live_vertex_confid,
		float4 *live_normal_radius,
		float4 *color_time,
		unsigned offset,
		mat34 cam2world)
	{
		const auto idx = threadIdx.x + blockDim.x * blockIdx.x;
		if (idx < surfel_array.Size())
		{
			const DepthSurfel &surfel = surfel_array[idx];
			auto vertex_world = cam2world.rot * surfel.vertex_confid + cam2world.trans;
			auto normal_world = cam2world.rot * surfel.normal_radius;
			reference_vertex_confid[idx + offset] = live_vertex_confid[idx + offset] = make_float4(vertex_world.x, vertex_world.y, vertex_world.z, surfel.vertex_confid.w);
			reference_normal_radius[idx + offset] = live_normal_radius[idx + offset] = make_float4(normal_world.x, normal_world.y, normal_world.z, surfel.normal_radius.w);
			color_time[idx + offset] = surfel.color_time;
		}
	}

	// For live geometry
	__global__ void initializerExpandDepthLiveSurfelKernel(
		GArrayView<DepthSurfel> surfel_array,
		float4 *live_vertex_confid,
		float4 *live_normal_radius,
		float4 *color_time,
		unsigned offset,
		mat34 cam2world)
	{
		const auto idx = threadIdx.x + blockDim.x * blockIdx.x;
		if (idx < surfel_array.Size())
		{
			const DepthSurfel &surfel = surfel_array[idx];
			auto vertex_world = cam2world.rot * surfel.vertex_confid + cam2world.trans;
			auto normal_world = cam2world.rot * surfel.normal_radius;
			live_vertex_confid[idx + offset] = make_float4(vertex_world.x, vertex_world.y, vertex_world.z, surfel.vertex_confid.w);
			live_normal_radius[idx + offset] = make_float4(normal_world.x, normal_world.y, normal_world.z, surfel.normal_radius.w);
			color_time[idx + offset] = surfel.color_time;
		}
	}

	// For cudaTexture
	__global__ void initializerSurfelKernel(
		cudaTextureObject_t vertex_confid_map,
		cudaTextureObject_t normal_radius_map,
		cudaTextureObject_t color_time_map,
		cudaTextureObject_t index_map,
		float4 *reference_vertex_confid,
		float4 *reference_normal_radius,
		float4 *live_vertex_confid,
		float4 *live_normal_radius,
		float4 *color_time,
		const mat34 cam2world,
		const unsigned img_rows,
		const unsigned img_cols)
	{
		const int x = threadIdx.x + blockDim.x * blockIdx.x;
		const int y = threadIdx.y + blockDim.y * blockIdx.y;
		if (x >= img_cols || y >= img_rows)
			return;

		float4 vertex_confid = tex2D<float4>(vertex_confid_map, x, y);
		if (fabs(vertex_confid.z) < 1e-6f)
			return; // Not valid

		unsigned index = tex2D<unsigned>(index_map, x, y);
		float3 vertex_confid_world = cam2world.rot * vertex_confid + cam2world.trans;
		vertex_confid = make_float4(
			vertex_confid_world.x,
			vertex_confid_world.y,
			vertex_confid_world.z,
			vertex_confid.w);
		reference_vertex_confid[index] = vertex_confid;
		live_vertex_confid[index] = vertex_confid;

		float4 normal_radius = tex2D<float4>(normal_radius_map, x, y);
		float3 normal_radius_world = cam2world.rot * normal_radius;
		normal_radius = make_float4(
			normal_radius_world.x,
			normal_radius_world.y,
			normal_radius_world.z,
			normal_radius.w);
		reference_normal_radius[index] = normal_radius;
		live_normal_radius[index] = normal_radius;

		float4 color_time_ = tex2D<float4>(color_time_map, x, y);
		color_time[index] = color_time_;
	}

	// For cudaTexture
	__global__ void initializerSurfelKernel(
		cudaTextureObject_t vertex_confid_map,
		cudaTextureObject_t normal_radius_map,
		cudaTextureObject_t color_time_map,
		cudaTextureObject_t index_map,
		cudaTextureObject_t segmentation_map,
		float4 *reference_vertex_confid,
		float4 *reference_normal_radius,
		float4 *live_vertex_confid,
		float4 *live_normal_radius,
		float4 *color_time,
		ucharX<d_max_num_semantic> *semantic_prob,
		const mat34 cam2world,
		const unsigned img_rows,
		const unsigned img_cols)
	{
		const int x = threadIdx.x + blockDim.x * blockIdx.x;
		const int y = threadIdx.y + blockDim.y * blockIdx.y;
		if (x >= img_cols || y >= img_rows)
			return;

		float4 vertex_confid = tex2D<float4>(vertex_confid_map, x, y);
		if (fabs(vertex_confid.z) < 1e-6f)
			return; // Not valid

		unsigned index = tex2D<unsigned>(index_map, x, y);
		float3 vertex_confid_world = cam2world.rot * vertex_confid + cam2world.trans;
		vertex_confid = make_float4(
			vertex_confid_world.x,
			vertex_confid_world.y,
			vertex_confid_world.z,
			vertex_confid.w);
		reference_vertex_confid[index] = vertex_confid;
		live_vertex_confid[index] = vertex_confid;

		float4 normal_radius = tex2D<float4>(normal_radius_map, x, y);
		float3 normal_radius_world = cam2world.rot * normal_radius;
		normal_radius = make_float4(
			normal_radius_world.x,
			normal_radius_world.y,
			normal_radius_world.z,
			normal_radius.w);
		reference_normal_radius[index] = normal_radius;
		live_normal_radius[index] = normal_radius;

		float4 color_time_ = tex2D<float4>(color_time_map, x, y);
		color_time[index] = color_time_;

		int semantic_label = tex2D<int>(segmentation_map, x, y);
		ucharX<d_max_num_semantic> semantic_prob_val;
		semantic_prob_val[semantic_label] = 2; // Semantic initialized as 2
		semantic_prob[index] = semantic_prob_val;
	}
};

void star::SurfelGeometryInitializer::InitFromObservationSerial(
	star::SurfelGeometry &geometry,
	const star::GArrayView<star::DepthSurfel> &surfel_array,
	cudaStream_t stream)
{
	geometry.ResizeValidSurfelArrays(surfel_array.Size());

	// Init the geometry
	const auto geometry_attributes = geometry.Geometry();
	initSurfelGeometry(geometry_attributes, surfel_array, stream);
}

void star::SurfelGeometryInitializer::InitFromMultiObservationSerial(
	SurfelGeometry &geometry,
	const unsigned num_cam,
	const GArrayView<DepthSurfel> *surfel_arrays,
	const Eigen::Matrix4f *cam2world,
	cudaStream_t stream)
{
	auto total_surfel_size = 0;
	for (auto i = 0; i < num_cam; ++i)
	{
		total_surfel_size += surfel_arrays[i].Size();
	}
	geometry.ResizeValidSurfelArrays(total_surfel_size);

	// Init the geometry
	const auto geometry_attributes = geometry.Geometry();
	initSurfelGeometry(geometry_attributes, num_cam, surfel_arrays, cam2world, stream);
}

void star::SurfelGeometryInitializer::InitFromMultiObservationSerial(
	LiveSurfelGeometry &geometry,
	const unsigned num_cam,
	const GArrayView<DepthSurfel> *surfel_arrays,
	const Eigen::Matrix4f *cam2world,
	cudaStream_t stream)
{
	auto total_surfel_size = 0;
	for (auto i = 0; i < num_cam; ++i)
	{
		total_surfel_size += surfel_arrays[i].Size();
	}

	geometry.ResizeValidSurfelArrays(total_surfel_size);

	// Init the geometry
	const auto geometry_attributes = geometry.Geometry();
	initSurfelGeometry(geometry_attributes, num_cam, surfel_arrays, cam2world, stream);
}

void star::SurfelGeometryInitializer::InitFromDataGeometry(
	SurfelGeometry &geometry,
	SurfelGeometry &data_geometry,
	const bool use_semantic,
	cudaStream_t stream)
{
	auto num_valid_surfels = data_geometry.NumValidSurfels();
	geometry.ResizeValidSurfelArrays(num_valid_surfels);
	auto geometry_attrib = geometry.Geometry();
	auto data_geometry_attrib = data_geometry.Geometry();
	// Do copy
	// TODO: change to parallel version
	cudaSafeCall(cudaMemcpyAsync(
		geometry_attrib.reference_vertex_confid,
		data_geometry_attrib.live_vertex_confid,
		sizeof(float4) * num_valid_surfels,
		cudaMemcpyDeviceToDevice,
		stream));
	cudaSafeCall(cudaMemcpyAsync(
		geometry_attrib.live_vertex_confid,
		data_geometry_attrib.live_vertex_confid,
		sizeof(float4) * num_valid_surfels,
		cudaMemcpyDeviceToDevice,
		stream));
	cudaSafeCall(cudaMemcpyAsync(
		geometry_attrib.reference_normal_radius,
		data_geometry_attrib.live_normal_radius,
		sizeof(float4) * num_valid_surfels,
		cudaMemcpyDeviceToDevice,
		stream));
	cudaSafeCall(cudaMemcpyAsync(
		geometry_attrib.live_normal_radius,
		data_geometry_attrib.live_normal_radius,
		sizeof(float4) * num_valid_surfels,
		cudaMemcpyDeviceToDevice,
		stream));
	cudaSafeCall(cudaMemcpyAsync(
		geometry_attrib.color_time,
		data_geometry_attrib.color_time,
		sizeof(float4) * num_valid_surfels,
		cudaMemcpyDeviceToDevice,
		stream));

	if (use_semantic)
	{
		// Optional
		cudaSafeCall(cudaMemcpyAsync(
			geometry_attrib.semantic_prob,
			data_geometry_attrib.semantic_prob,
			sizeof(ucharX<d_max_num_semantic>) * num_valid_surfels,
			cudaMemcpyDeviceToDevice,
			stream));
	}
	// Resize
	geometry.ResizeValidSurfelArrays(num_valid_surfels);
}

void star::SurfelGeometryInitializer::initSurfelGeometry(
	GeometryAttributes geometry,
	const SurfelMap &surfel_map,
	const Eigen::Matrix4f &cam2world,
	cudaStream_t stream)
{
	unsigned width, height;
	query2DTextureExtent(surfel_map.VertexConfidReadOnly(), width, height);

	dim3 blk(16, 16);
	dim3 grid(divUp(width, blk.x), divUp(height, blk.y));
	device::initializerSurfelKernel<<<grid, blk, 0, stream>>>(
		surfel_map.VertexConfidReadOnly(),
		surfel_map.NormalRadiusReadOnly(),
		surfel_map.ColorTimeReadOnly(),
		surfel_map.IndexReadOnly(),
		geometry.reference_vertex_confid,
		geometry.reference_normal_radius,
		geometry.live_vertex_confid,
		geometry.live_normal_radius,
		geometry.color_time,
		mat34(cam2world),
		height,
		width);
}

void star::SurfelGeometryInitializer::initSurfelGeometry(
	GeometryAttributes geometry,
	const GArrayView<DepthSurfel> &surfel_array,
	cudaStream_t stream)
{
	// Invoke the kernel
	dim3 blk(256);
	dim3 grid(divUp(surfel_array.Size(), blk.x));
	device::initializerCollectDepthSurfelKernel<<<grid, blk, 0, stream>>>(
		surfel_array,
		geometry.reference_vertex_confid,
		geometry.reference_normal_radius,
		geometry.live_vertex_confid,
		geometry.live_normal_radius,
		geometry.color_time);

	// Sync and check error
#if defined(CUDA_DEBUG_SYNC_CHECK)
	cudaSafeCall(cudaStreamSynchronize(stream));
	cudaSafeCall(cudaGetLastError());
#endif
}

void star::SurfelGeometryInitializer::initSurfelGeometry(
	GeometryAttributes geometry,
	const unsigned num_cam,
	const GArrayView<DepthSurfel> *surfel_arrays,
	const Eigen::Matrix4f *cam2world,
	cudaStream_t stream)
{
	// Invoke the kernel
	unsigned offset = 0;
	dim3 blk(256);
	for (auto i = 0; i < num_cam; ++i)
	{
		mat34 cam2world_mat(cam2world[i]);
		dim3 grid(divUp(surfel_arrays[i].Size(), blk.x));
		device::initializerExpandDepthSurfelKernel<<<grid, blk, 0, stream>>>(
			surfel_arrays[i],
			geometry.reference_vertex_confid,
			geometry.reference_normal_radius,
			geometry.live_vertex_confid,
			geometry.live_normal_radius,
			geometry.color_time,
			offset, // offset
			cam2world_mat);
		offset += surfel_arrays[i].Size();
	}

	// Sync and check error
#if defined(CUDA_DEBUG_SYNC_CHECK)
	cudaSafeCall(cudaStreamSynchronize(stream));
	cudaSafeCall(cudaGetLastError());
#endif
}

void star::SurfelGeometryInitializer::initSurfelGeometry(
	LiveGeometryAttributes geometry,
	const unsigned num_cam,
	const GArrayView<DepthSurfel> *surfel_arrays,
	const Eigen::Matrix4f *cam2world,
	cudaStream_t stream)
{
	// Invoke the kernel
	unsigned offset = 0;
	dim3 blk(256);
	for (auto i = 0; i < num_cam; ++i)
	{
		mat34 cam2world_mat(cam2world[i]);
		dim3 grid(divUp(surfel_arrays[i].Size(), blk.x));
		device::initializerExpandDepthLiveSurfelKernel<<<grid, blk, 0, stream>>>(
			surfel_arrays[i],
			geometry.live_vertex_confid,
			geometry.live_normal_radius,
			geometry.color_time,
			offset, // offset
			cam2world_mat);
		offset += surfel_arrays[i].Size();
	}

	// Sync and check error
#if defined(CUDA_DEBUG_SYNC_CHECK)
	cudaSafeCall(cudaStreamSynchronize(stream));
	cudaSafeCall(cudaGetLastError());
#endif
}

void star::SurfelGeometryInitializer::InitFromGeometryMap(
	SurfelGeometry &geometry,
	const SurfelMap &surfel_map,
	const Eigen::Matrix4f &cam2world,
	cudaStream_t stream)
{
	geometry.ResizeValidSurfelArrays(surfel_map.NumValidSurfels());

	// Init the geometry
	const auto geometry_attributes = geometry.Geometry();
	initSurfelGeometry(geometry_attributes, surfel_map, cam2world, stream);
}