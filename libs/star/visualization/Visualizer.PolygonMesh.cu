#include <pcl/Vertices.h>
#include <pcl/PolygonMesh.h>
#include <pcl/PCLPointCloud2.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/conversions.h>
#include <pcl/io/obj_io.h>
#include <star/common/data_transfer.h>
#include <star/visualization/Visualizer.h>

void star::visualize::SavePolygonMesh(
    const GArrayView<float4> &vertex,
    const GArrayView<int> &faces, // Triangle
    const std::string &path)
{
    auto polygon = std::make_shared<pcl::PolygonMesh>();

    // Cloud
    auto point_cloud = downloadPointCloud(GArray<float4>((float4 *)vertex.Ptr(), vertex.Size()));
    pcl::toPCLPointCloud2(*point_cloud, polygon->cloud);

    // Vertices
    std::vector<int> h_faces;
    faces.Download(h_faces);

    for (auto i = 0; i < h_faces.size(); i += 3)
    {
        pcl::Vertices vertices;
        vertices.vertices.push_back(
            h_faces[i]);
        vertices.vertices.push_back(
            h_faces[i + 1]);
        vertices.vertices.push_back(
            h_faces[i + 2]);
        polygon->polygons.push_back(vertices);
    }

    // Save
    pcl::io::saveOBJFile(path, *polygon);
}