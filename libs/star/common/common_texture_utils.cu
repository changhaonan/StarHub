#include <star/common/common_texture_utils.h>

cudaTextureObject_t star::create1DLinearTexture(const GArray<float> &array)
{
	cudaTextureDesc texture_desc;
	memset(&texture_desc, 0, sizeof(cudaTextureDesc));
	texture_desc.normalizedCoords = 0;
	texture_desc.addressMode[0] = cudaAddressModeBorder; // Return 0 outside the boundary
	texture_desc.addressMode[1] = cudaAddressModeBorder;
	texture_desc.addressMode[2] = cudaAddressModeBorder;
	texture_desc.filterMode = cudaFilterModePoint;
	texture_desc.readMode = cudaReadModeElementType;
	texture_desc.sRGB = 0;

	// Create resource desc
	cudaResourceDesc resource_desc;
	memset(&resource_desc, 0, sizeof(cudaResourceDesc));
	resource_desc.resType = cudaResourceTypeLinear;
	resource_desc.res.linear.devPtr = (void *)array.ptr();
	resource_desc.res.linear.sizeInBytes = array.sizeBytes();
	resource_desc.res.linear.desc.f = cudaChannelFormatKindFloat;
	resource_desc.res.linear.desc.x = 32;
	resource_desc.res.linear.desc.y = 0;
	resource_desc.res.linear.desc.z = 0;
	resource_desc.res.linear.desc.w = 0;

	// Allocate the texture
	cudaTextureObject_t d_texture;
	cudaSafeCall(cudaCreateTextureObject(&d_texture, &resource_desc, &texture_desc, nullptr));
	return d_texture;
}

cudaTextureObject_t star::create1DLinearTexture(const GBufferArray<float> &array)
{
	GArray<float> pcl_array((float *)array.Ptr(), array.BufferSize());
	return create1DLinearTexture(pcl_array);
}

void star::createDefault2DTextureDesc(cudaTextureDesc &desc)
{
	memset(&desc, 0, sizeof(desc));
	desc.addressMode[0] = cudaAddressModeBorder; // Return 0 outside the boundary
	desc.addressMode[1] = cudaAddressModeBorder;
	desc.addressMode[2] = cudaAddressModeBorder;
	desc.filterMode = cudaFilterModePoint;
	desc.readMode = cudaReadModeElementType;
	desc.normalizedCoords = 0;
}

void star::createDefault3DTextureDesc(cudaTextureDesc &desc)
{
	memset(&desc, 0, sizeof(desc));
	desc.addressMode[0] = cudaAddressModeBorder; // Return 0 outside the boundary
	desc.addressMode[1] = cudaAddressModeBorder;
	desc.addressMode[2] = cudaAddressModeBorder;
	desc.filterMode = cudaFilterModePoint;
	desc.readMode = cudaReadModeElementType;
	desc.normalizedCoords = 0;
}

void star::createDepthTexture(
	const unsigned img_rows,
	const unsigned img_cols,
	cudaTextureObject_t &texture,
	cudaArray_t &d_array)
{
	// The texture description
	cudaTextureDesc depth_texture_desc;
	createDefault2DTextureDesc(depth_texture_desc);

	// Create channel descriptions
	cudaChannelFormatDesc depth_channel_desc = cudaCreateChannelDesc(16, 0, 0, 0, cudaChannelFormatKindUnsigned);

	// Allocate the cuda array
	cudaSafeCall(cudaMallocArray(&d_array, &depth_channel_desc, img_cols, img_rows));

	// Create the resource desc
	cudaResourceDesc resource_desc;
	memset(&resource_desc, 0, sizeof(cudaResourceDesc));
	resource_desc.resType = cudaResourceTypeArray;
	resource_desc.res.array.array = d_array;

	// Allocate the texture
	cudaSafeCall(cudaCreateTextureObject(&texture, &resource_desc, &depth_texture_desc, 0));
}

void star::createDepthTextureSurface(
	const unsigned img_rows,
	const unsigned img_cols,
	cudaTextureObject_t &texture,
	cudaSurfaceObject_t &surface,
	cudaArray_t &d_array)
{
	// The texture description
	cudaTextureDesc depth_texture_desc;
	createDefault2DTextureDesc(depth_texture_desc);

	// Create channel descriptions
	cudaChannelFormatDesc depth_channel_desc = cudaCreateChannelDesc(16, 0, 0, 0, cudaChannelFormatKindUnsigned);

	// Allocate the cuda array
	cudaSafeCall(cudaMallocArray(&d_array, &depth_channel_desc, img_cols, img_rows));

	// Create the resource desc
	cudaResourceDesc resource_desc;
	memset(&resource_desc, 0, sizeof(cudaResourceDesc));
	resource_desc.resType = cudaResourceTypeArray;
	resource_desc.res.array.array = d_array;

	// Allocate the texture
	cudaSafeCall(cudaCreateTextureObject(&texture, &resource_desc, &depth_texture_desc, 0));
	cudaSafeCall(cudaCreateSurfaceObject(&surface, &resource_desc));
}

void star::createDepthTextureSurface(const unsigned img_rows, const unsigned img_cols, CudaTextureSurface &collect)
{
	STAR_CHECK_NE(img_rows, 0);
	STAR_CHECK_NE(img_cols, 0);
	createDepthTextureSurface(
		img_rows, img_cols,
		collect.texture, collect.surface, collect.d_array);
}

void star::createIndexTextureSurface(
	const unsigned img_rows,
	const unsigned img_cols,
	cudaTextureObject_t &texture,
	cudaSurfaceObject_t &surface,
	cudaArray_t &d_array)
{
	// The texture description
	cudaTextureDesc index_texture_desc;
	createDefault2DTextureDesc(index_texture_desc);

	// Create channel descriptions
	cudaChannelFormatDesc index_channel_desc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindUnsigned);

	// Allocate the cuda array
	cudaSafeCall(cudaMallocArray(&d_array, &index_channel_desc, img_cols, img_rows));

	// Create the resource desc
	cudaResourceDesc resource_desc;
	memset(&resource_desc, 0, sizeof(cudaResourceDesc));
	resource_desc.resType = cudaResourceTypeArray;
	resource_desc.res.array.array = d_array;

	// Allocate the texture
	cudaSafeCall(cudaCreateTextureObject(&texture, &resource_desc, &index_texture_desc, 0));
	cudaSafeCall(cudaCreateSurfaceObject(&surface, &resource_desc));
}

void star::createIndexTextureSurface(const unsigned img_rows, const unsigned img_cols, CudaTextureSurface &collect)
{
	STAR_CHECK_NE(img_rows, 0);
	STAR_CHECK_NE(img_cols, 0);
	createIndexTextureSurface(
		img_rows, img_cols,
		collect.texture, collect.surface, collect.d_array);
}

void star::createInt32TextureSurface(
	const unsigned img_rows, const unsigned img_cols,
	cudaTextureObject_t &texture, cudaSurfaceObject_t &surface,
	cudaArray_t &d_array)
{
	// The texture description
	cudaTextureDesc int32_texture_desc;
	createDefault2DTextureDesc(int32_texture_desc);

	// Create channel descriptions
	cudaChannelFormatDesc int32_channel_desc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindSigned);

	// Allocate the cuda array
	cudaSafeCall(cudaMallocArray(&d_array, &int32_channel_desc, img_cols, img_rows));

	// Create the resource desc
	cudaResourceDesc resource_desc;
	memset(&resource_desc, 0, sizeof(cudaResourceDesc));
	resource_desc.resType = cudaResourceTypeArray;
	resource_desc.res.array.array = d_array;

	// Allocate the texture
	cudaSafeCall(cudaCreateTextureObject(&texture, &resource_desc, &int32_texture_desc, 0));
	cudaSafeCall(cudaCreateSurfaceObject(&surface, &resource_desc));
}

void star::createInt32TextureSurface(
	const unsigned img_rows, const unsigned img_cols,
	CudaTextureSurface &collect)
{
	STAR_CHECK_NE(img_rows, 0);
	STAR_CHECK_NE(img_cols, 0);
	createInt32TextureSurface(
		img_rows, img_cols,
		collect.texture, collect.surface, collect.d_array);
}

void star::createFloat4TextureSurface(
	const unsigned rows, const unsigned cols,
	cudaTextureObject_t &texture,
	cudaSurfaceObject_t &surface,
	cudaArray_t &d_array)
{
	// The texture description
	cudaTextureDesc float4_texture_desc;
	createDefault2DTextureDesc(float4_texture_desc);

	// Create channel descriptions
	cudaChannelFormatDesc float4_channel_desc = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);

	// Allocate the cuda array
	cudaSafeCall(cudaMallocArray(&d_array, &float4_channel_desc, cols, rows));

	// Create the resource desc
	cudaResourceDesc resource_desc;
	memset(&resource_desc, 0, sizeof(cudaResourceDesc));
	resource_desc.resType = cudaResourceTypeArray;
	resource_desc.res.array.array = d_array;

	// Allocate the texture
	cudaSafeCall(cudaCreateTextureObject(&texture, &resource_desc, &float4_texture_desc, 0));
	cudaSafeCall(cudaCreateSurfaceObject(&surface, &resource_desc));
}

void star::createFloat4TextureSurface(const unsigned rows, const unsigned cols, CudaTextureSurface &texture_collect)
{
	STAR_CHECK_NE(rows, 0);
	STAR_CHECK_NE(cols, 0);
	createFloat4TextureSurface(
		rows, cols,
		texture_collect.texture,
		texture_collect.surface,
		texture_collect.d_array);
}

void star::createFloat3TextureSurface(
	const unsigned rows, const unsigned cols,
	cudaTextureObject_t &texture,
	cudaSurfaceObject_t &surface,
	cudaArray_t &d_array)
{
	// The texture description
	cudaTextureDesc float3_texture_desc;
	createDefault2DTextureDesc(float3_texture_desc);

	// Create channel descriptions
	cudaChannelFormatDesc float3_channel_desc = cudaCreateChannelDesc(32, 32, 32, 0, cudaChannelFormatKindFloat);

	// Allocate the cuda array
	cudaSafeCall(cudaMallocArray(&d_array, &float3_channel_desc, cols, rows));

	// Create the resource desc
	cudaResourceDesc resource_desc;
	memset(&resource_desc, 0, sizeof(cudaResourceDesc));
	resource_desc.resType = cudaResourceTypeArray;
	resource_desc.res.array.array = d_array;

	// Allocate the texture
	cudaSafeCall(cudaCreateTextureObject(&texture, &resource_desc, &float3_texture_desc, 0));
	cudaSafeCall(cudaCreateSurfaceObject(&surface, &resource_desc));
}

void star::createFloat3TextureSurface(
	const unsigned rows, const unsigned cols,
	CudaTextureSurface &texture_collect)
{
	STAR_CHECK_NE(rows, 0);
	STAR_CHECK_NE(cols, 0);
	createFloat3TextureSurface(
		rows, cols,
		texture_collect.texture,
		texture_collect.surface,
		texture_collect.d_array);
}

void star::createFloat2TextureSurface(
	const unsigned rows, const unsigned cols,
	cudaTextureObject_t &texture,
	cudaSurfaceObject_t &surface,
	cudaArray_t &d_array)
{
	// The texture description
	cudaTextureDesc float2_texture_desc;
	createDefault2DTextureDesc(float2_texture_desc);

	// Create channel descriptions
	cudaChannelFormatDesc float2_channel_desc = cudaCreateChannelDesc(32, 32, 0, 0, cudaChannelFormatKindFloat);

	// Allocate the cuda array
	cudaSafeCall(cudaMallocArray(&d_array, &float2_channel_desc, cols, rows));

	// Create the resource desc
	cudaResourceDesc resource_desc;
	memset(&resource_desc, 0, sizeof(cudaResourceDesc));
	resource_desc.resType = cudaResourceTypeArray;
	resource_desc.res.array.array = d_array;

	// Allocate the texture
	cudaSafeCall(cudaCreateTextureObject(&texture, &resource_desc, &float2_texture_desc, 0));
	cudaSafeCall(cudaCreateSurfaceObject(&surface, &resource_desc));
}

void star::createFloat2TextureSurface(
	const unsigned rows, const unsigned cols,
	CudaTextureSurface &texture_collect)
{
	STAR_CHECK_NE(rows, 0);
	STAR_CHECK_NE(cols, 0);
	createFloat2TextureSurface(
		rows, cols,
		texture_collect.texture,
		texture_collect.surface,
		texture_collect.d_array);
}

void star::createFloat1TextureSurface(
	const unsigned rows, const unsigned cols,
	cudaTextureObject_t &texture,
	cudaSurfaceObject_t &surface,
	cudaArray_t &d_array,
	const bool interpolationable)
{
	// The texture description
	cudaTextureDesc float1_texture_desc;
	createDefault2DTextureDesc(float1_texture_desc);

	if (interpolationable)
	{
		float1_texture_desc.filterMode = cudaFilterModeLinear;
	}
	else
	{
		// Deafault point
	}

	// Create channel descriptions
	cudaChannelFormatDesc float1_channel_desc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);

	// Allocate the cuda array
	cudaSafeCall(cudaMallocArray(&d_array, &float1_channel_desc, cols, rows));

	// Create the resource desc
	cudaResourceDesc resource_desc;
	memset(&resource_desc, 0, sizeof(cudaResourceDesc));
	resource_desc.resType = cudaResourceTypeArray;
	resource_desc.res.array.array = d_array;

	// Allocate the texture
	cudaSafeCall(cudaCreateTextureObject(&texture, &resource_desc, &float1_texture_desc, 0));
	cudaSafeCall(cudaCreateSurfaceObject(&surface, &resource_desc));
}

void star::createFloat1TextureSurface(
	const unsigned rows, const unsigned cols,
	CudaTextureSurface &texture_collect,
	const bool interpolationable)
{
	STAR_CHECK_NE(rows, 0);
	STAR_CHECK_NE(cols, 0);
	createFloat1TextureSurface(
		rows, cols,
		texture_collect.texture,
		texture_collect.surface,
		texture_collect.d_array,
		interpolationable);
}

void star::createUChar1TextureSurface(
	const unsigned rows, const unsigned cols,
	cudaTextureObject_t &texture,
	cudaSurfaceObject_t &surface,
	cudaArray_t &d_array)
{
	// The texture description
	cudaTextureDesc uchar1_texture_desc;
	createDefault2DTextureDesc(uchar1_texture_desc);

	// Create channel descriptions
	cudaChannelFormatDesc uchar1_channel_desc = cudaCreateChannelDesc(8, 0, 0, 0, cudaChannelFormatKindUnsigned);

	// Allocate the cuda array
	cudaSafeCall(cudaMallocArray(&d_array, &uchar1_channel_desc, cols, rows));

	// Create the resource desc
	cudaResourceDesc resource_desc;
	memset(&resource_desc, 0, sizeof(cudaResourceDesc));
	resource_desc.resType = cudaResourceTypeArray;
	resource_desc.res.array.array = d_array;

	// Allocate the texture
	cudaSafeCall(cudaCreateTextureObject(&texture, &resource_desc, &uchar1_texture_desc, 0));
	cudaSafeCall(cudaCreateSurfaceObject(&surface, &resource_desc));
}

void star::createUChar1TextureSurface(
	const unsigned rows, const unsigned cols,
	CudaTextureSurface &texture_collect)
{
	STAR_CHECK_NE(rows, 0);
	STAR_CHECK_NE(cols, 0);
	createUChar1TextureSurface(
		rows, cols,
		texture_collect.texture,
		texture_collect.surface,
		texture_collect.d_array);
}

void star::query2DTextureExtent(cudaTextureObject_t texture, unsigned &width, unsigned &height)
{
	cudaResourceDesc texture_res;
	cudaSafeCall(cudaGetTextureObjectResourceDesc(&texture_res, texture));
	cudaArray_t cu_array = texture_res.res.array.array;
	cudaChannelFormatDesc channel_desc;
	cudaExtent extent;
	unsigned int flag;
	cudaSafeCall(cudaArrayGetInfo(&channel_desc, &extent, &flag, cu_array));

	width = extent.width;
	height = extent.height;
}

void star::releaseTextureCollect(CudaTextureSurface &texture_collect)
{
	cudaSafeCall(cudaDestroyTextureObject(texture_collect.texture));
	cudaSafeCall(cudaDestroySurfaceObject(texture_collect.surface));
	cudaSafeCall(cudaFreeArray(texture_collect.d_array));
}

void star::createFloat3DTextureSurface(
	const unsigned cols, const unsigned rows, const unsigned height,
	cudaTextureObject_t &texture,
	cudaSurfaceObject_t &surface,
	cudaArray_t &d_array,
	const bool interpolationable)
{

	// The texture description
	cudaTextureDesc float1_texture_desc;
	createDefault3DTextureDesc(float1_texture_desc);

	if (interpolationable)
	{
		float1_texture_desc.filterMode = cudaFilterModeLinear;
	}
	else
	{
		// Deafault point
	}

	// Create channel descriptions
	cudaChannelFormatDesc float1_channel_desc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);

	// Allocate the cuda array
	cudaSafeCall(cudaMalloc3DArray(&d_array, &float1_channel_desc, make_cudaExtent(cols, rows, height)));

	// Create the resource desc
	cudaResourceDesc resource_desc;
	memset(&resource_desc, 0, sizeof(cudaResourceDesc));
	resource_desc.resType = cudaResourceTypeArray;
	resource_desc.res.array.array = d_array;

	// Allocate the texture
	cudaSafeCall(cudaCreateTextureObject(&texture, &resource_desc, &float1_texture_desc, 0));
	cudaSafeCall(cudaCreateSurfaceObject(&surface, &resource_desc));
}

void star::createFloat3DTextureSurface(
	const unsigned cols, const unsigned rows, const unsigned height,
	CudaTextureSurface &texture_collect,
	const bool interpolationable)
{
	STAR_CHECK_NE(rows, 0);
	STAR_CHECK_NE(cols, 0);
	STAR_CHECK_NE(height, 0);
	createFloat3DTextureSurface(
		cols, rows, height,
		texture_collect.texture,
		texture_collect.surface,
		texture_collect.d_array,
		interpolationable);
}

void star::createUchar3DTextureSurface(
	const unsigned cols, const unsigned rows, const unsigned height,
	cudaTextureObject_t &texture,
	cudaSurfaceObject_t &surface,
	cudaArray_t &d_array)
{
	// The texture description
	cudaTextureDesc uchar_texture_desc;
	createDefault3DTextureDesc(uchar_texture_desc);

	// Create channel descriptions
	cudaChannelFormatDesc uchar_channel_desc = cudaCreateChannelDesc(8, 0, 0, 0, cudaChannelFormatKindUnsigned);

	// Allocate the cuda array
	cudaSafeCall(cudaMalloc3DArray(&d_array, &uchar_channel_desc, make_cudaExtent(cols, rows, height)));

	// Create the resource desc
	cudaResourceDesc resource_desc;
	memset(&resource_desc, 0, sizeof(cudaResourceDesc));
	resource_desc.resType = cudaResourceTypeArray;
	resource_desc.res.array.array = d_array;

	// Allocate the texture
	cudaSafeCall(cudaCreateTextureObject(&texture, &resource_desc, &uchar_texture_desc, 0));
	cudaSafeCall(cudaCreateSurfaceObject(&surface, &resource_desc));
}

void star::createUchar3DTextureSurface(
	const unsigned cols, const unsigned rows, const unsigned height,
	CudaTextureSurface &texture_collect)
{
	STAR_CHECK_NE(rows, 0);
	STAR_CHECK_NE(cols, 0);
	STAR_CHECK_NE(height, 0);
	createUchar3DTextureSurface(
		cols, rows, height,
		texture_collect.texture,
		texture_collect.surface,
		texture_collect.d_array);
}

void star::query3DTextureExtent(
	cudaTextureObject_t texture, unsigned &width, unsigned &height, unsigned &depth)
{
	cudaResourceDesc texture_res;
	cudaSafeCall(cudaGetTextureObjectResourceDesc(&texture_res, texture));
	cudaArray_t cu_array = texture_res.res.array.array;
	cudaChannelFormatDesc channel_desc;
	cudaExtent extent;
	unsigned int flag;
	cudaSafeCall(cudaArrayGetInfo(&channel_desc, &extent, &flag, cu_array));

	width = extent.width;
	height = extent.height;
	depth = extent.depth;
}