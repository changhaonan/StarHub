#pragma once
#include <star/common/macro_utils.h>
#include <star/common/common_types.h>
#include <star/common/surfel_types.h>
#include <star/common/ArrayView.h>
#include <star/common/types/typeX.h>
#include <star/common/point_cloud_typedefs.h>
#include <star/visualization/Visualizer.Host.h> // Host visualization functions
#include <opencv2/opencv.hpp>
#include <memory>

namespace star::visualize
{
	/**
	 * \brief This module is for DEBUG visualization. Any methods
	 *        in this module should NOT be used in real-time code.
	 */

	/* Constants
	 */
	extern const std::map<int, uchar3> default_semantic_color_dict;

	/* The depth image drawing methods
	 */
	void DrawDepthImage(const cv::Mat &depth_img);
	void SaveDepthImage(const cv::Mat &depth_img, const std::string &path);
	void DrawDepthImage(const GArray2D<unsigned short> &depth_img);
	void SaveDepthImage(const GArray2D<unsigned short> &depth_img, const std::string &path);
	void DrawDepthImage(cudaTextureObject_t depth_img);
	void SaveDepthImage(cudaTextureObject_t depth_img, const std::string &path);
	void DrawDepthFloatImage(cudaTextureObject_t depth_img); // Draw Depth Image in float format

	/* The color image drawing methods
	 */
	void DrawRGBImage(const cv::Mat &rgb_img);
	void SaveRGBImage(const cv::Mat &rgb_img, const std::string &path);
	void DrawRGBImage(const GArray<uchar3> &rgb_img, const int rows, const int cols);
	void SaveRGBImage(const GArray<uchar3> &rgb_img, const int rows, const int cols, const std::string &path);
	void DrawNormalizeRGBImage(cudaTextureObject_t rgb_img);
	void SaveNormalizeRGBImage(cudaTextureObject_t rgb_img, const std::string &path);
	void DrawNormalizeRGBDImage(cudaTextureObject_t rgbd_img);
	void SaveNormalizeRGBDImage(cudaTextureObject_t rgbd_img, const std::string &rgb_path, const std::string &depth_path);
	void DrawColorTimeMap(cudaTextureObject_t color_time_map);
	void SaveColorTimeMap(cudaTextureObject_t color_time_map, const std::string &path);
	void DrawNormalMap(cudaTextureObject_t normal_map);

	/* The semantic image drawing method
	 */
	// semantic_img is of CV_32SC1 (int32)
	void DrawSemanticImage(const cv::Mat &semantic_img, const std::map<int, uchar3> &color_dict);
	void DrawSemanticMap(cudaTextureObject_t semantic_map, const std::map<int, uchar3> &color_dict);
	void SaveSemanticImage(const cv::Mat &semantic_img, const std::map<int, uchar3> &color_dict, const std::string &path);
	void SaveSemanticMap(cudaTextureObject_t semantic_map, const std::map<int, uchar3> &color_dict, const std::string &path);

	/* The gray scale image drawing for filtered
	 */
	void DrawGrayScaleImage(const cv::Mat &gray_scale_img);
	void SaveGrayScaleImage(const cv::Mat &gray_scale_img, const std::string &path);
	void DrawGrayScaleImage(cudaTextureObject_t gray_scale_img, float scale = 1.0f);
	void SaveGrayScaleImage(cudaTextureObject_t gray_scale_img, const std::string &path, float scale = 1.0f);

	/* The segmentation mask drawing methods
	 */
	void MarkSegmentationMask(
		const std::vector<unsigned char> &mask,
		cv::Mat &rgb_img,
		const unsigned sample_rate = 2);
	void DrawSegmentMask(
		const std::vector<unsigned char> &mask,
		cv::Mat &rgb_img,
		const unsigned sample_rate = 2);
	void SaveSegmentMask(
		const std::vector<unsigned char> &mask,
		cv::Mat &rgb_img,
		const std::string &path,
		const unsigned sample_rate = 2);
	void DrawSegmentMask(
		cudaTextureObject_t mask,
		cudaTextureObject_t normalized_rgb_img,
		const unsigned sample_rate = 2);
	void SaveSegmentMask(
		cudaTextureObject_t mask,
		cudaTextureObject_t normalized_rgb_img,
		const std::string &path,
		const unsigned sample_rate = 2);
	void SaveRawSegmentMask(
		cudaTextureObject_t mask,
		const std::string &path);
	void DrawRawSegmentMask(
		cudaTextureObject_t mask);

	/*  Save sampling image pixels
	 */
	void SaveSampledRGBImage(
		cudaTextureObject_t rgb_img,
		GArrayView<unsigned> sampled_indicator,
		const std::string &path);

	/* The binary meanfield drawing methods
	 */
	void DrawBinaryMeanfield(cudaTextureObject_t meanfield_q);
	void SaveBinaryMeanfield(cudaTextureObject_t meanfield_q, const std::string &path);

	/* Visualize the valid geometry maps as binary mask
	 * If validity_halfsize is smaller than 0, the validity only depends on the pixel,
	 * otherwise, we will search within a window of validity_halfsize.
	 */
	void DrawValidIndexMap(cudaTextureObject_t index_map, int validity_halfsize);
	void SaveValidIndexMap(cudaTextureObject_t index_map, int validity_halfsize, const std::string &path);

	cv::Mat GetValidityMapCV(cudaTextureObject_t index_map, int validity_halfsize);
	// Mark the validity of each index map pixel and save them to flatten indicator
	// Assume pre-allcoated indicator
	void MarkValidIndexMapValue(
		cudaTextureObject_t index_map,
		int validity_halfsize,
		GArray<unsigned char> flatten_validity_indicator);

	/* Visualize indicator
	 */
	void DrawIndicatorMap(
		const GArrayView<unsigned> &indicator,
		size_t img_height,
		size_t img_width);
	void SaveIndicatorMap(
		const GArrayView<unsigned> &indicator,
		size_t img_height,
		size_t img_width,
		const std::string &img_path);
	cv::Mat GetIndicatorMapCV(
		const GArrayView<unsigned> &indicator,
		size_t img_height,
		size_t img_width);

	/*
	 * Visualize 2d pixel
	 */
	void DrawPixel2DMap(
		const GArrayView<ushort2> &pixel_2d,
		size_t img_height,
		size_t img_width);
	void SavePixel2DMap(
		const GArrayView<ushort2> &pixel_2d,
		size_t img_height,
		size_t img_width,
		const std::string &img_path);
	cv::Mat GetPixel2DMapCV(
		const GArrayView<ushort2> &pixel_2d,
		size_t img_height,
		size_t img_width);

	/*
	 * Visualize optical flow
	 */
	void DrawOpticalFlowMap(cudaTextureObject_t opticalflow);
	void SaveOpticalFlowMap(cudaTextureObject_t opticalflow, const std::string &path);
	/* The correspondence
	 */
	void DrawImagePairCorrespondence(
		cudaTextureObject_t rgb_0, cudaTextureObject_t rgb_1,
		const GArray<ushort4> &correspondence);

	/* The point cloud drawing methods
	 */
	void DrawPointCloud(const GArray<float4> &point_cloud);
	void DrawPointCloud(const GArrayView<float4> &point_cloud);
	void DrawPointCloud(const GArray2D<float4> &vertex_map);
	void DrawPointCloud(const GArray<DepthSurfel> &surfel_array);
	void DrawPointCloud(cudaTextureObject_t vertex_map);
	void SavePointCloud(const std::vector<float4> &point_cloud, const std::string &path);
	void SavePointCloud(cudaTextureObject_t veretx_map, const std::string &path);
	void SavePointCloud(const GArrayView<float4> point_cloud, const std::string &path);
	void SavePointCloud(const GArray<float4> &point_cloud, const std::string &path);

	/* The point cloud with normal
	 */
	void DrawPointCloudWithNormal(
		const PointCloud3f_Pointer &point_cloud
#ifdef WITH_PCL
		,
		const PointCloudNormal_Pointer &normal_cloud
#endif
	);
	void DrawPointCloudWithNormal(
		const GArray<float4> &vertex_cloud,
		const GArray<float4> &normal_cloud);
	void DrawPointCloudWithNormal(
		const GArrayView<float4> &vertex_cloud,
		const GArrayView<float4> &normal_cloud);
	void DrawPointCloudWithNormal(
		const GArray2D<float4> &vertex_map,
		const GArray2D<float4> &normal_map);
	void DrawPointCloudWithNormal(cudaTextureObject_t vertex_map, cudaTextureObject_t normal_map);
	void DrawPointCloudWithNormal(const GArray<DepthSurfel> &surfel_array);
	void SavePointCloudWithNormal(cudaTextureObject_t vertex_map, cudaTextureObject_t normal_map); // Save in ofstream
	void SavePointCloudWithNormal(cudaTextureObject_t vertex_map, cudaTextureObject_t normal_map, const std::string &path); // Save in ofstream
	void SavePointCloudWithNormal(const GArrayView<float4> &vertex, const GArrayView<float4> &normal, const std::string &path);

	/* The colored point cloud
	 */
	void DrawColoredPointCloud(const PointCloud3fRGB_Pointer &point_cloud);
	void SaveColoredPointCloud(const PointCloud3fRGB_Pointer &point_cloud, const std::string &path);
	void DrawColoredPointCloud(const GArray<float4> &vertex, const GArray<float4> &color_time);
	void DrawColoredPointCloud(const GArrayView<float4> &vertex, const GArrayView<float4> &color_time);
	void DrawColoredPointCloud(cudaTextureObject_t vertex_map, cudaTextureObject_t color_time_map);
	void SaveColoredPointCloud(cudaTextureObject_t vertex_map, cudaTextureObject_t color_time_map, const std::string &path);
	void SaveColoredPointCloud(const GArrayView<float4> &vertex, const GArrayView<float4> &color_time, const std::string &path);

	/* The colored point normal cloud
	 */
	void SaveColoredPointCloudWithNormal(
		const GArrayView<float4> &vertex, const GArrayView<float4> &color_time, const GArrayView<float4> &normal, const std::string &path);

	/* The heated map point cloud
	 */
	void SaveHeatedPointCloud(cudaTextureObject_t vertex_map, cudaTextureObject_t heat_map, const std::string &path, const float scale = 10.0f);
	void SaveHeatedPointCloud(const GArrayView<float4> &vertex_array, const GArrayView<unsigned> &heat_array, const std::string &path, const float scale = 10.0f);
	void SaveHeatedPointCloud(const GArrayView<float4> &vertex_array, const GArrayView<float> &heat_array, const std::string &path, const float scale = 10.0f);
	/* The heated map point cloud with normal
	 */
	void SaveHeatedPointCloudWithNormal(
		const GArrayView<float4> &vertex_array,
		const GArrayView<float4> &normal_array,
		const GArrayView<unsigned> &heat_array,
		const std::string &path,
		const float scale = 10.0f,
		const std::string &color_map = "jet");

	/* The semantic point cloud
	 */
	template <unsigned num_semantic>
	void SaveSemanticPointCloud(
		const GArrayView<float4> &vertex,
		const GArrayView<ucharX<num_semantic>> &semantic_prob,
		const std::map<int, uchar3> &color_dict,
		const std::string &path);

	template <unsigned num_semantic>
	void SaveSegmentationPointCloud(
		cudaTextureObject_t vertex_confid_map,
		cudaTextureObject_t segmentation_map,
		const std::map<int, uchar3> &color_dict,
		const std::string &path);

	/* The confidence of point cloud
	 */
	void SaveConfidencePointCloud(const GArrayView<float4> &vertex_confid, const std::string &path, const float scale = 0.1f);
	void SaveConfidencePointCloud(cudaTextureObject_t vertex_confid_map, const std::string &path, const float scale = 0.1f);

	/* The time map of point cloud
	 */
	void SaveTimePointCloud(
		const GArrayView<float4> &vertex,
		const GArrayView<float4> &color_time,
		const std::string &path,
		const float current_time);

	/* The polygon mesh
	 */
	void SavePolygonMesh(
		const GArrayView<float4> &vertex,
		const GArrayView<int> &faces, // Triangle
		const std::string &path);

	/* The matched point cloud
	 */
	void DrawMatchedCloudPair(const PointCloud3f_Pointer &cloud_1,
							  const PointCloud3f_Pointer &cloud_2);
	void DrawMatchedCloudPair(const PointCloud3f_Pointer &cloud_1,
							  const PointCloud3f_Pointer &cloud_2,
							  const Eigen::Matrix4f &from1To2);
	void DrawMatchedCloudPair(
		cudaTextureObject_t cloud_1,
		const GArray<float4> &cloud_2,
		const Matrix4f &from1To2);
	void DrawMatchedCloudPair(
		cudaTextureObject_t cloud_1,
		const GArrayView<float4> &cloud_2,
		const Matrix4f &from1To2);
	void DrawMatchedCloudPair(cudaTextureObject_t cloud_1,
							  cudaTextureObject_t cloud_2,
							  const Matrix4f &from1To2);

	void SaveMatchedCloudPair(
		const PointCloud3f_Pointer &cloud_1,
		const PointCloud3f_Pointer &cloud_2,
		const std::string &cloud_1_name, const std::string &cloud_2_name);
	void SaveMatchedCloudPair(
		const PointCloud3f_Pointer &cloud_1,
		const PointCloud3f_Pointer &cloud_2,
		const Eigen::Matrix4f &from1To2,
		const std::string &cloud_1_name, const std::string &cloud_2_name);
	void SaveMatchedCloudPair(
		cudaTextureObject_t cloud_1,
		const GArray<float4> &cloud_2,
		const Matrix4f &from1To2,
		const std::string &cloud_1_name, const std::string &cloud_2_name);
	void SaveMatchedCloudPair(
		cudaTextureObject_t cloud_1,
		const GArrayView<float4> &cloud_2,
		const Matrix4f &from1To2,
		const std::string &cloud_1_name, const std::string &cloud_2_name);

	/* The method to draw matched color-point cloud
	 */
	void DrawMatchedRGBCloudPair(const PointCloud3fRGB_Pointer &cloud_1,
								 const PointCloud3fRGB_Pointer &cloud_2);
	void DrawMatchedRGBCloudPair(const PointCloud3fRGB_Pointer &cloud_1,
								 const PointCloud3fRGB_Pointer &cloud_2,
								 const Eigen::Matrix4f &from1To2);
	void DrawMatchedCloudPair(
		cudaTextureObject_t vertex_map, cudaTextureObject_t color_time_map,
		const GArrayView<float4> &surfel_array, const GArrayView<float4> &color_time_array,
		const Eigen::Matrix4f &camera2world);

	// Matched PointCloud
	void SaveMatchedPointCloud(
		const GArrayView<float4> &src_vertex_confid,
		const GArrayView<float4> &tar_vertex_confid,
		const std::string &path);

	/* The method to draw fused point cloud
	 */
	void DrawFusedSurfelCloud(
		GArrayView<float4> surfel_vertex,
		GArrayView<unsigned> fused_indicator);
	void DrawFusedSurfelCloud(
		GArrayView<float4> surfel_vertex,
		unsigned num_remaining_surfels);

	void DrawFusedAppendedSurfelCloud(
		GArrayView<float4> surfel_vertex,
		GArrayView<unsigned> fused_indicator,
		cudaTextureObject_t depth_vertex_map,
		GArrayView<unsigned> append_indicator,
		const Matrix4f &world2camera);

	void DrawAppendedSurfelCloud(
		GArrayView<float4> surfel_vertex,
		cudaTextureObject_t depth_vertex_map,
		GArrayView<unsigned> append_indicator,
		const Matrix4f &world2camera);
	void DrawAppendedSurfelCloud(
		GArrayView<float4> surfel_vertex,
		cudaTextureObject_t depth_vertex_map,
		GArrayView<ushort2> append_pixel,
		const Matrix4f &world2camera);

	template <typename TPointInput, typename TNormalsInput>
	void DrawPointCloudWithNormals_Generic(TPointInput &points, TNormalsInput &normals);

	// Supporting ushort4
	void SaveGraph(
		const GArrayView<float4> &vertices,
		const GArrayView<ushort4> &edges,
		const std::string &path);

	void SaveGraph(
		const GArrayView<float4> &vertices,
		const GArrayView<float> &vertex_weight,
		const GArrayView<ushort4> &edges,
		const std::string &path);

	void SaveGraph( // Unsigned variant of the previous
		const GArrayView<float4> &vertices,
		const GArrayView<unsigned> &vertex_weight,
		const GArrayView<ushort4> &edges,
		const std::string &path);

	void SaveGraph(
		const GArrayView<float4> &vertices,
		const GArrayView<float4> &normals,
		const GArrayView<ushort4> &edges,
		const std::string &path);

	void SaveGraph(
		const GArrayView<float4> &vertices,
		const GArrayView<float4> &normals,
		const GArrayView<ushort4> &edges,
		const GArrayView<float4> &edge_weight,
		const std::string &path);

	void SaveGraph(
		const GArrayView<float4> &vertices,
		const GArrayView<ushort4> &edges,
		const GArrayView<float4> &edge_weight,
		const std::string &path);

	void SaveGraph(
		const GArrayView<float4> &vertices,
		const GArrayView<uchar3> &vertex_color,
		const GArrayView<ushort4> &edges,
		const GArrayView<float4> &edge_weight,
		const std::string &path);

	// All in One
	void SaveGraph_Generic(
		const GArrayView<float4> &vertices,
		const GArrayView<uchar3> &vertex_color,
		const GArrayView<float> &vertex_weight,
		const GArrayView<float4> &normals,
		const GArrayView<float> &normal_weight,
		const GArrayView<ushort4> &edges,
		const GArrayView<float4> &edge_weight,
		const std::string &path);

	// All in one unsigned variant
	// TODO: Change it to template version? If needed.
	void SaveGraph_Generic(
		const GArrayView<float4> &vertices,
		const GArrayView<uchar3> &vertex_color,
		const GArrayView<unsigned> &vertex_weight,
		const GArrayView<float4> &normals,
		const GArrayView<float> &normal_weight,
		const GArrayView<ushort4> &edges,
		const GArrayView<float4> &edge_weight,
		const std::string &path);

	// FIXME: Some graph visualizer has been put to the header file, not sure why they can't be defined here

	/* The method for color transfer
	 */
	template <unsigned num_semantic>
	inline void Semantic2Color(
		const GArrayView<ucharX<num_semantic>> &semantic_prob_array,
		const std::map<int, uchar3> &color_dict,
		std::vector<uchar3> &output_color_array);
};

// Header-implementation
#include <star/visualization/Visualizer.Semantic.cuh>
#include <star/visualization/Visualizer.Graph.cuh>