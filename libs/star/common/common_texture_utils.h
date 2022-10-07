#pragma once

#include <star/common/common_types.h>
#include <star/common/GBufferArray.h>
#include <cuda_texture_types.h>
#include <exception>
#include <cuda.h>

namespace star
{

	/**
	 * \brief Create of 1d linear float texturem, accessed by fetch1DLinear.
	 *        Using the array as the underline memory
	 */
	cudaTextureObject_t create1DLinearTexture(const GArray<float> &array);
	cudaTextureObject_t create1DLinearTexture(const GBufferArray<float> &array);

	/**
	 * \brief Create TextureDesc for default 2D texture
	 */
	void createDefault2DTextureDesc(cudaTextureDesc &desc);

	/**
	 * \brief Create TextureDesc for default 3D texture
	 */
	void createDefault3DTextureDesc(cudaTextureDesc &desc);

	/**
	 * \brief Create 2D uint16 textures (and surfaces) for depth image
	 */
	void createDepthTexture(
		const unsigned img_rows, const unsigned img_cols,
		cudaTextureObject_t &texture, cudaArray_t &d_array);
	void createDepthTextureSurface(
		const unsigned img_rows, const unsigned img_cols,
		cudaTextureObject_t &texture, cudaSurfaceObject_t &surface,
		cudaArray_t &d_array);
	void createDepthTextureSurface(
		const unsigned img_rows, const unsigned img_cols,
		CudaTextureSurface &collect);

	/**
	 * \brief Create 2D uint32 textures (and surfaces) for index map
	 */
	void createIndexTextureSurface(
		const unsigned img_rows, const unsigned img_cols,
		cudaTextureObject_t &texture, cudaSurfaceObject_t &surface,
		cudaArray_t &d_array);
	void createIndexTextureSurface(
		const unsigned img_rows, const unsigned img_cols,
		CudaTextureSurface &collect);

	/**
	 * \brief Create 2D int32 textures (and surfaces) for index map
	 */
	void createInt32TextureSurface(
		const unsigned img_rows, const unsigned img_cols,
		cudaTextureObject_t &texture, cudaSurfaceObject_t &surface,
		cudaArray_t &d_array);
	void createInt32TextureSurface(
		const unsigned img_rows, const unsigned img_cols,
		CudaTextureSurface &collect);

	/**
	 * \brief Create 2D float4 textures (and surfaces) for all kinds of use
	 */
	void createFloat4TextureSurface(
		const unsigned rows, const unsigned cols,
		cudaTextureObject_t &texture, cudaSurfaceObject_t &surface,
		cudaArray_t &d_array);
	void createFloat4TextureSurface(
		const unsigned rows, const unsigned cols,
		CudaTextureSurface &texture_collect);

	/**
	 * \brief Create 2D float2 textures (and surfaces) for gradient map
	 */
	void createFloat2TextureSurface(
		const unsigned rows, const unsigned cols,
		cudaTextureObject_t &texture, cudaSurfaceObject_t &surface,
		cudaArray_t &d_array);
	void createFloat2TextureSurface(
		const unsigned rows, const unsigned cols,
		CudaTextureSurface &texture_collect);

	/**
	 * \brief Create 2D float3 textures (and surfaces) for gradient map
	 */
	void createFloat3TextureSurface(
		const unsigned rows, const unsigned cols,
		cudaTextureObject_t &texture, cudaSurfaceObject_t &surface,
		cudaArray_t &d_array);
	void createFloat3TextureSurface(
		const unsigned rows, const unsigned cols,
		CudaTextureSurface &texture_collect);

	/**
	 * \brief Create 2D float1 textures (and surfaces) for mean-field inference & depth & otherthing
	 */
	void createFloat1TextureSurface(
		const unsigned rows, const unsigned cols,
		cudaTextureObject_t &texture, cudaSurfaceObject_t &surface,
		cudaArray_t &d_array,
		const bool interpolationable);
	void createFloat1TextureSurface(
		const unsigned rows, const unsigned cols,
		CudaTextureSurface &texture_collect,
		const bool interpolationable = false);

	/**
	 * \brief Create 2D uchar1 textures (and surfaces) for binary mask
	 */
	void createUChar1TextureSurface(
		const unsigned rows, const unsigned cols,
		cudaTextureObject_t &texture, cudaSurfaceObject_t &surface,
		cudaArray_t &d_array);
	void createUChar1TextureSurface(
		const unsigned rows, const unsigned cols,
		CudaTextureSurface &texture_collect);

	/**
	 * \brief Release 2D texture
	 */
	void releaseTextureCollect(CudaTextureSurface &texture_collect);

	/**
	 * \brief The query functions for 2D texture
	 */
	void query2DTextureExtent(cudaTextureObject_t texture, unsigned &width, unsigned &height);

	/**
	 * \brief Create 16-bit float 3D texture (and surfaces)
	 */
	void createFloat3DTextureSurface(
		const unsigned cols, const unsigned rows, const unsigned height,
		cudaTextureObject_t &texture,
		cudaSurfaceObject_t &surface,
		cudaArray_t &d_array,
		const bool interpolationable);
	void createFloat3DTextureSurface(
		const unsigned cols, const unsigned rows, const unsigned height,
		CudaTextureSurface &texture_collect, const bool interpolationable = false);

	/**
	 * \brief Create uchar3 3D texture (and surfaces)
	 */
	void createUchar3DTextureSurface(
		const unsigned cols, const unsigned rows, const unsigned height,
		cudaTextureObject_t &texture,
		cudaSurfaceObject_t &surface,
		cudaArray_t &d_array);
	void createUchar3DTextureSurface(
		const unsigned cols, const unsigned rows, const unsigned height,
		CudaTextureSurface &texture_collect);

	/**
	 * \brief The query function for 3D texture
	 */
	void query3DTextureExtent(
		cudaTextureObject_t texture, unsigned &width, unsigned &height, unsigned &depth);

} // namespace star