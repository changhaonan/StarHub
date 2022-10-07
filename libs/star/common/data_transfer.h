#pragma once
#include <star/common/common_types.h>
#include <star/common/surfel_types.h>
#include <star/common/ArrayView.h>
#include <star/common/point_cloud_typedefs.h>
#include <opencv2/opencv.hpp>
#include <memory>

namespace star
{
	/* Download the image from GPU memory to CPU memory
	 */
	cv::Mat downloadDepthImage(const GArray2D<unsigned short> &image_gpu);
	cv::Mat downloadDepthImage(cudaTextureObject_t image_gpu);
	cv::Mat downloadDepthFloatImage(const GArray2D<float> &image_gpu);
	cv::Mat downloadDepthFloatImage(cudaTextureObject_t image_gpu);
	cv::Mat downloadOptcalFlowImage(cudaTextureObject_t image_gpu);
	cv::Mat downloadRGBImage(
		const GArray<uchar3> &image_gpu,
		const unsigned rows, const unsigned cols);
	cv::Mat downloadSemanticImage(cudaTextureObject_t image_gpu); // Semantic image is int32

	// The rgb texture is in float4
	cv::Mat downloadNormalizeRGBImage(const GArray2D<float4> &rgb_img);
	cv::Mat downloadNormalizeRGBImage(cudaTextureObject_t rgb_img);
	cv::Mat rgbImageFromColorTimeMap(cudaTextureObject_t color_time_map);
	cv::Mat normalMapForVisualize(cudaTextureObject_t normal_map);

	// The segmentation mask texture
	void downloadSegmentationMask(cudaTextureObject_t mask, std::vector<unsigned char> &h_mask);
	cv::Mat downloadRawSegmentationMask(cudaTextureObject_t mask); // uchar texture

	// The gray scale image
	void downloadGrayScaleImage(cudaTextureObject_t image, cv::Mat &h_image, float scale = 1.0f);

	// The binary meanfield map, the texture contains the
	// mean field probability of the positive label
	void downloadTransferBinaryMeanfield(cudaTextureObject_t meanfield_q, cv::Mat &h_meanfield_uchar);

	/* The point cloud download functions
	 */
	PointCloud3f_Pointer downloadPointCloud(const GArray<float4> &vertex);
	PointCloud3f_Pointer downloadPointCloud(const GArray2D<float4> &vertex_map);
	PointCloud3f_Pointer downloadPointCloud(const GArray2D<float4> &vertex_map, GArrayView<unsigned> indicator);
	PointCloud3f_Pointer downloadPointCloud(const GArray2D<float4> &vertex_map, GArrayView<ushort2> pixel);
	void downloadPointCloud(const GArray2D<float4> &vertex_map, std::vector<float4> &point_cloud);
	PointCloud3f_Pointer downloadPointCloud(cudaTextureObject_t vertex_map);
	PointCloud3f_Pointer downloadPointCloud(cudaTextureObject_t vertex_map, GArrayView<unsigned> indicator);
	PointCloud3f_Pointer downloadPointCloud(cudaTextureObject_t vertex_map, GArrayView<ushort2> pixel);
	void downloadPointCloud(cudaTextureObject_t vertex_map, std::vector<float4> &point_cloud);

	void downloadPointNormalCloud(
		const GArray<DepthSurfel> &surfel_array,
		PointCloud3f_Pointer &point_cloud,
		PointCloudNormal_Pointer &normal_cloud,
		const float point_scale = 1.0f);

	// Download it with indicator
	void separateDownloadPointCloud(
		const GArrayView<float4> &point_cloud,
		const GArrayView<unsigned> &indicator,
		PointCloud3f_Pointer &fused_cloud,
		PointCloud3f_Pointer &unfused_cloud);
	void separateDownloadPointCloud(
		const GArrayView<float4> &point_cloud,
		unsigned num_remaining_surfels,
		PointCloud3f_Pointer &remaining_cloud,
		PointCloud3f_Pointer &appended_cloud);

	/* The normal cloud download functions
	 */
	PointCloudNormal_Pointer downloadNormalCloud(const GArray<float4> &normal_cloud);
	PointCloudNormal_Pointer downloadNormalCloud(const GArray2D<float4> &normal_map);
	PointCloudNormal_Pointer downloadNormalCloud(cudaTextureObject_t normal_map);

	/* The colored point cloud download function
	 */
	PointCloud3fRGB_Pointer downloadColoredPointCloud(
		const GArray<float4> &vertex_confid,
		const GArray<float4> &color_time,
		bool flip_color = false);
	PointCloud3fRGB_Pointer downloadColoredPointCloud(
		cudaTextureObject_t vertex_map,
		cudaTextureObject_t color_time_map,
		bool flip_color = false);

	PointCloud3fRGB_Pointer downloadHeatedPointCloud(
		cudaTextureObject_t vertex_map,
		cudaTextureObject_t heat_map,
		const float scale);
	PointCloud3fRGB_Pointer downloadHeatedPointCloud(
		const GArrayView<float4> &vertex_array,
		const GArrayView<unsigned> &heat_array,
		const float scale,
		const std::string &color_map = "jet");
	PointCloud3fRGB_Pointer downloadHeatedPointCloud(
		const GArrayView<float4> &vertex_array,
		const GArrayView<float> &heat_array,
		const float scale);

	template <unsigned num_semantic>
	PointCloud3fRGB_Pointer downloadSemanticPointCloud(
		cudaTextureObject_t vertex_confid_map,
		cudaTextureObject_t segmentation_map,
		const std::map<int, uchar3> &color_dict);

	template <unsigned num_semantic>
	PointCloud3fRGB_Pointer downloadSemanticPointCloud(
		const GArrayView<float4> &vertex_array,
		const GArrayView<ucharX<num_semantic>> &semantic_prob,
		const std::map<int, uchar3> &color_dict);

	PointCloud3fRGB_Pointer downloadConfidencePointCloud(
		cudaTextureObject_t vertex_confid_map,
		const float scale = 0.1f);
	PointCloud3fRGB_Pointer downloadConfidencePointCloud(
		const GArrayView<float4> &vertex_confid,
		const float scale = 0.1f);

	PointCloud3fRGB_Pointer downloadTimePointCloud(
		const GArrayView<float4> &vertex_confid,
		const GArrayView<float4> &color_time,
		const float current_time);
	/* Colorize the point cloud
	 */
	PointCloud3fRGB_Pointer addColorToPointCloud(const PointCloud3f_Pointer &point_cloud, uchar4 rgba);

	/* Query the index map
	 */
	void queryIndexMapFromPixels(cudaTextureObject_t index_map, const GArrayView<ushort4> &pixel_array, GArray<unsigned> &knn_array);

	/* Transfer the memory from texture to GPU memory.
	 * Assume ALLOCATED device memory.
	 */
	template <typename T>
	void textureToMap2D(
		cudaTextureObject_t texture,
		GArray2D<T> &map,
		cudaStream_t stream = 0);

	/* ArrayViewAs, Parse ArrayView<T1> as ArrayView<T2>
	 */
	template <typename T1, typename T2>
	GArrayView<T2> ArrayViewCast(
		const GArrayView<T1> &array_view_t1)
	{
		return GArrayView<T2>(
			(T2 *)array_view_t1.Ptr(),
			array_view_t1.ByteSize() / sizeof(T2));
	}

	/* Color transfer related
	 */
	template <unsigned num_semantic>
	void semanticTocolor(
		const GArrayView<ucharX<num_semantic>> &semantic_prob_array,
		const std::map<int, uchar3> &color_dict,
		std::vector<uchar3> &output_color_array);
}

#if defined(__CUDACC__)
#include <star/common/data_transfer.cuh>
#endif