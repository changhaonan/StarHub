#pragma once
#include <pcl/io/pcd_io.h>
#include "Visualizer.h"

// Too many data, so we need to sample
template <unsigned knn_size>
void star::visualize::SaveValidSkinning(
    const GArrayView<float4> &surfel_vertex,
    const GArrayView<float4> &node_vertex,
    const GArrayView<ushortX<knn_size>> &skinning,
    const GArrayView<floatX<knn_size>> &skinning_connect,
    const float valid_threshold,
    const float sample_rate,
    const std::string &path)
{
    STAR_CHECK_EQ(surfel_vertex.Size(), skinning.Size());
    STAR_CHECK_EQ(surfel_vertex.Size(), skinning_connect.Size());
    
    GArray<float4> surfel_vertex_array((float4 *)surfel_vertex.Ptr(), surfel_vertex.Size());
    const auto surfel_vertex_cloud = downloadPointCloud(surfel_vertex_array);
    GArray<float4> node_vertex_array((float4 *)node_vertex.Ptr(), node_vertex.Size());
    const auto node_vertex_cloud = downloadPointCloud(node_vertex_array);

    std::vector<ushortX<knn_size>> skinning_array;
    std::vector<floatX<knn_size>> skinning_connect_array;
    skinning.Download(skinning_array);
    skinning_connect.Download(skinning_connect_array);
    // Build
    unsigned num_valid_skinning = 0;
    PointNormalCloud3f skinning_point_cloud;
    float sample_counter;
    for (auto i = 0; i < skinning_array.size(); ++i) {
        const auto &skinning_i = skinning_array[i];
        const auto &skinning_connect_i = skinning_connect_array[i];
        sample_counter += sample_rate;
        if (sample_counter < 1.0f) {
            continue;
        }
        else {
            sample_counter -= 1.0f;
        }
        for (auto j = 0; j < knn_size; ++j) {
            if (skinning_connect_i[j] > valid_threshold) {
                pcl::PointNormal p;
                p.x = surfel_vertex_cloud->points[i].x;
                p.y = surfel_vertex_cloud->points[i].y;
                p.z = surfel_vertex_cloud->points[i].z;
                p.normal_x = node_vertex_cloud->points[skinning_i[j]].x - surfel_vertex_cloud->points[i].x;
                p.normal_y = node_vertex_cloud->points[skinning_i[j]].y - surfel_vertex_cloud->points[i].y;
                p.normal_z = node_vertex_cloud->points[skinning_i[j]].z - surfel_vertex_cloud->points[i].z;

                skinning_point_cloud.points.push_back(p);
                ++num_valid_skinning;
            }
        }
    }
    skinning_point_cloud.resize(num_valid_skinning);
    pcl::io::savePCDFileBinary(path, skinning_point_cloud);
}