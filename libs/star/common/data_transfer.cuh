#pragma once
#include <star/common/common_types.h>
#include <star/common/common_utils.h>
#include <star/common/logging.h>
#include <star/common/data_transfer.h>
#include <star/common/common_point_cloud_utils.h>
#include <star/common/types/vecX_op.h>
#include <star/common/common_texture_utils.h>
#include <device_launch_parameters.h>

namespace star
{
	namespace device
	{

		template <typename T>
		__global__ void textureToMap2DKernel(
			cudaTextureObject_t texture,
			PtrStepSz<T> map)
		{
			const auto x = threadIdx.x + blockDim.x * blockIdx.x;
			const auto y = threadIdx.y + blockDim.y * blockIdx.y;
			if (x >= map.cols || y >= map.rows)
				return;
			T element = tex2D<T>(texture, x, y);
			map.ptr(y)[x] = element;
		}

	}; /* End of namespace device */
};	   /* End of namespace star */

template <typename T>
void star::textureToMap2D(
	cudaTextureObject_t texture,
	GArray2D<T> &map,
	cudaStream_t stream)
{
	dim3 blk(16, 16);
	dim3 grid(divUp(map.cols(), blk.x), divUp(map.rows(), blk.y));
	device::textureToMap2DKernel<T><<<grid, blk, 0, stream>>>(texture, map);
}

/* PCD-related
 */
template <unsigned num_semantic>
PointCloud3fRGB_Pointer star::downloadSemanticPointCloud(
	const GArrayView<float4> &vertex_array,
	const GArrayView<ucharX<num_semantic>> &semantic_prob,
	const std::map<int, uchar3> &color_dict)
{
	// Download it
	std::vector<float4> h_vertex_array;
	std::vector<ucharX<num_semantic>> h_semantic_prob;
	vertex_array.Download(h_vertex_array);
	semantic_prob.Download(h_semantic_prob);

	// Set point_cloud
	int num_points = h_vertex_array.size();
	PointCloud3fRGB_Pointer point_cloud(new PointCloud3fRGB());
	setPointCloudRGBSize(point_cloud, num_points);
	for (auto idx = 0; idx < num_points; idx++)
	{
		auto label = max_id(h_semantic_prob[idx]);
		uchar3 rgb = color_dict.at(label);
		setPointRGB(h_vertex_array[idx].x, h_vertex_array[idx].y, h_vertex_array[idx].z,
					rgb.x, rgb.y, rgb.z,
					point_cloud, idx);
	}

	return point_cloud;
}

template <unsigned num_semantic>
PointCloud3fRGB_Pointer star::downloadSemanticPointCloud(
	cudaTextureObject_t vertex_confid_map,
	cudaTextureObject_t segmentation_map,
	const std::map<int, uchar3> &color_dict)
{
	// Pre-check
	unsigned vertex_cols, vertex_rows;
	query2DTextureExtent(vertex_confid_map, vertex_cols, vertex_rows);
	unsigned seg_cols, seg_rows;
	query2DTextureExtent(segmentation_map, seg_cols, seg_rows);
	STAR_CHECK_EQ(seg_cols, vertex_cols);
	STAR_CHECK_EQ(seg_rows, vertex_rows);

	GArray2D<float4> vertex_confid_array;
	vertex_confid_array.create(vertex_rows, vertex_cols);
	textureToMap2D<float4>(vertex_confid_map, vertex_confid_array);
	GArray2D<int> segmentation_array;
	segmentation_array.create(vertex_rows, vertex_cols);
	textureToMap2D<int>(segmentation_map, segmentation_array);

	// Download it
	float4 *h_vertex = new float4[vertex_rows * vertex_cols];
	vertex_confid_array.download(h_vertex, vertex_cols * sizeof(float4));
	int *h_segmentation = new int[vertex_rows * vertex_cols];
	segmentation_array.download(h_segmentation, vertex_cols * sizeof(int));

	// Set point_cloud
	int num_points = vertex_rows * vertex_cols;
	PointCloud3fRGB_Pointer point_cloud(new PointCloud3fRGB());
	setPointCloudRGBSize(point_cloud, num_points);
	for (auto idx = 0; idx < num_points; idx++)
	{
		int label = h_segmentation[idx];
		uchar3 rgb = color_dict.at(label);
		setPointRGB(
			h_vertex[idx].x, h_vertex[idx].y, h_vertex[idx].z,
			rgb.x, rgb.y, rgb.z,
			point_cloud, idx);
	}

	return point_cloud;
}

template <unsigned num_semantic>
void star::semanticTocolor(
	const GArrayView<ucharX<num_semantic>> &semantic_prob_array,
	const std::map<int, uchar3> &color_dict,
	std::vector<uchar3> &output_color_array)
{
	// 1. Download it
	std::vector<ucharX<num_semantic>> h_semantic_prob_array;
	semantic_prob_array.Download(h_semantic_prob_array);

	// 2. Transfer to color
	output_color_array.clear();
	output_color_array.reserve(h_semantic_prob_array.size());

	for (auto i = 0; i < h_semantic_prob_array.size(); ++i)
	{
		auto label = max_id(h_semantic_prob_array[i]);
		uchar3 rgb = color_dict.at(label);
		output_color_array.emplace_back(rgb);
	}
}