#pragma once
#include <star/common/data_transfer.h>
#include <star/visualization/Visualizer.h>

template <unsigned num_semantic>
void star::visualize::SaveSemanticPointCloud(
	const GArrayView<float4> &vertex,
	const GArrayView<ucharX<num_semantic>> &semantic_prob,
	const std::map<int, uchar3> &color_dict,
	const std::string &path)
{
	auto cloud = downloadSemanticPointCloud<num_semantic>(
		vertex, semantic_prob, color_dict);
	SaveColoredPointCloud(cloud, path);
}

template <unsigned num_semantic>
void star::visualize::SaveSegmentationPointCloud(
	cudaTextureObject_t vertex_confid_map,
	cudaTextureObject_t segmentation_map,
	const std::map<int, uchar3> &color_dict,
	const std::string &path)
{
	auto cloud = downloadSemanticPointCloud<num_semantic>(
		vertex_confid_map, segmentation_map, color_dict);
	SaveColoredPointCloud(cloud, path);
}

template <unsigned num_semantic>
void star::visualize::Semantic2Color(
	const GArrayView<ucharX<num_semantic>> &semantic_prob_array,
	const std::map<int, uchar3> &color_dict,
	std::vector<uchar3> &output_color_array)
{
	semanticTocolor(semantic_prob_array, color_dict, output_color_array);
}