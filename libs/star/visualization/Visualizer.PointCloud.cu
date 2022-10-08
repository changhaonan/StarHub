#include <star/common/Stream.h>
#include <star/common/Serializer.h>
#include <star/common/BinaryFileStream.h>
#include <star/common/data_transfer.h>
#include <star/common/common_point_cloud_utils.h>
#include <star/visualization/Visualizer.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/obj_io.h>

/* The point cloud drawing methods
 */
void star::visualize::DrawPointCloud(const star::GArray<float4> &point_cloud)
{
    const auto h_point_cloud = downloadPointCloud(point_cloud);
    DrawPointCloud(h_point_cloud);
}

void star::visualize::DrawPointCloud(const star::GArrayView<float4> &cloud)
{
    GArray<float4> cloud_array = GArray<float4>((float4 *)cloud.Ptr(), cloud.Size());
    DrawPointCloud(cloud_array);
}

void star::visualize::DrawPointCloud(const GArray2D<float4> &vertex_map)
{
    const auto point_cloud = downloadPointCloud(vertex_map);
    DrawPointCloud(point_cloud);
}

void star::visualize::DrawPointCloud(cudaTextureObject_t vertex_map)
{
    const auto point_cloud = downloadPointCloud(vertex_map);
    DrawPointCloud(point_cloud);
}

void star::visualize::DrawPointCloud(
    const star::GArray<star::DepthSurfel> &surfel_array)
{
    PointCloud3f_Pointer point_cloud;
    PointCloudNormal_Pointer normal_cloud;
    downloadPointNormalCloud(surfel_array, point_cloud, normal_cloud);
    DrawPointCloud(point_cloud);
}

void star::visualize::SavePointCloud(const std::vector<float4> &point_vec, const std::string &path)
{
    std::ofstream file_output;
    file_output.open(path);
    file_output << "OFF" << std::endl;
    file_output << point_vec.size() << " " << 0 << " " << 0 << std::endl;
    for (int node_iter = 0; node_iter < point_vec.size(); node_iter++)
    {
        file_output << point_vec[node_iter].x
                    << " " << point_vec[node_iter].y << " "
                    << point_vec[node_iter].z
                    << std::endl;
    }
}

void star::visualize::SavePointCloud(cudaTextureObject_t vertex_map, const std::string &path)
{
    PointCloud3f_Pointer point_cloud = downloadPointCloud(vertex_map);
    point_cloud->width = 1;
    point_cloud->height = point_cloud->points.size();
    pcl::io::savePCDFileASCII(path, *point_cloud);
}

void star::visualize::SavePointCloud(const GArrayView<float4> cloud, const std::string &path)
{
    GArray<float4> point_cloud((float4 *)cloud.Ptr(), cloud.Size());
    SavePointCloud(point_cloud, path);
}

void star::visualize::SavePointCloud(const GArray<float4> &cloud, const std::string &path)
{
    PointCloud3f_Pointer point_cloud = downloadPointCloud(cloud);
    point_cloud->width = 1;
    point_cloud->height = point_cloud->points.size();
    pcl::io::savePCDFileASCII(path, *point_cloud);
}

/* The point cloud with normal
 */
void star::visualize::DrawPointCloudWithNormal(
    const PointCloud3f_Pointer &point_cloud,
    const PointCloudNormal_Pointer &normal_cloud)
{
    const std::string window_title = "3D Viewer";
    // boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer(window_title));
    // viewer->setBackgroundColor(0, 0, 0);
    // pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> handler(point_cloud, 0, 255, 0);
    // viewer->addPointCloud<pcl::PointXYZ>(point_cloud, handler, "sample cloud");
    // viewer->addPointCloudNormals<pcl::PointXYZ, pcl::Normal>(point_cloud, normal_cloud, 30, 60.0f);
    // while (!viewer->wasStopped()) {
    //     viewer->spinOnce(100);
    // }
}

template <typename TPointInput, typename TNormalsInput>
void star::visualize::DrawPointCloudWithNormals_Generic(TPointInput &points, TNormalsInput &normals)
{
    const auto point_cloud = downloadPointCloud(points);
    const auto normal_cloud = downloadNormalCloud(normals);
    DrawPointCloudWithNormal(point_cloud, normal_cloud);
}

void star::visualize::DrawPointCloudWithNormal(
    const GArray<float4> &vertex,
    const GArray<float4> &normal)
{
    DrawPointCloudWithNormals_Generic(vertex, normal);
}

void star::visualize::DrawPointCloudWithNormal(
    const GArrayView<float4> &vertex_cloud,
    const GArrayView<float4> &normal_cloud)
{
    STAR_CHECK(vertex_cloud.Size() == normal_cloud.Size());
    GArray<float4> vertex_array((float4 *)vertex_cloud.Ptr(), vertex_cloud.Size());
    GArray<float4> normal_array((float4 *)normal_cloud.Ptr(), normal_cloud.Size());
    DrawPointCloudWithNormal(vertex_array, normal_array);
}

void star::visualize::DrawPointCloudWithNormal(
    const GArray2D<float4> &vertex_map,
    const GArray2D<float4> &normal_map)
{
    DrawPointCloudWithNormals_Generic(vertex_map, normal_map);
}

void star::visualize::DrawPointCloudWithNormal(
    cudaTextureObject_t vertex_map,
    cudaTextureObject_t normal_map)
{
    DrawPointCloudWithNormals_Generic(vertex_map, normal_map);
}

void star::visualize::DrawPointCloudWithNormal(
    const GArray<DepthSurfel> &surfel_array)
{
    PointCloud3f_Pointer point_cloud;
    PointCloudNormal_Pointer normal_cloud;
    downloadPointNormalCloud(surfel_array, point_cloud, normal_cloud);
    DrawPointCloudWithNormal(point_cloud, normal_cloud);
}

void star::visualize::SavePointCloudWithNormal(cudaTextureObject_t vertex_map, cudaTextureObject_t normal_map)
{
    // Download it
    const auto point_cloud = downloadPointCloud(vertex_map);
    const auto normal_cloud = downloadNormalCloud(normal_map);

    // Construct the output stream
    BinaryFileStream output_fstream("pointnormal", BinaryFileStream::Mode::WriteOnly);

    // Prepare the test data
    std::vector<float4> save_vec;
    for (auto i = 0; i < point_cloud->points.size(); i++)
    {
        save_vec.push_back(
            make_float4(point_cloud->points[i].x, point_cloud->points[i].y, point_cloud->points[i].z, 0));
        save_vec.push_back(make_float4(
            normal_cloud->points[i].normal_x,
            normal_cloud->points[i].normal_y,
            normal_cloud->points[i].normal_z,
            0));
    }

    // Save it
    // PODVectorSerializeHandler<int>::Write(&output_fstream, save_vec);
    // SerializeHandler<std::vector<int>>::Write(&output_fstream, save_vec);
    // output_fstream.Write<std::vector<int>>(save_vec);
    output_fstream.SerializeWrite<std::vector<float4>>(save_vec);
}

void star::visualize::SavePointCloudWithNormal(
    const GArrayView<float4> &vertex,
    const GArrayView<float4> &normal,
    const std::string &path)
{
    GArray<float4> vertex_array((float4 *)vertex.Ptr(), vertex.Size());
    GArray<float4> normal_array((float4 *)normal.Ptr(), normal.Size());
    const auto point_cloud = downloadPointCloud(vertex_array);
    const auto normal_cloud = downloadNormalCloud(normal_array);

    // Concate point and normal
    PointNormalCloud3f_Pointer point_normal_cloud = boost::make_shared<PointNormalCloud3f>();
    pcl::concatenateFields(*point_cloud, *normal_cloud, *point_normal_cloud);
    // redefine pcd size
    point_normal_cloud->width = 1;
    point_normal_cloud->height = point_normal_cloud->points.size();
    pcl::io::savePCDFileASCII(path, *point_normal_cloud);
}

/* The colored point cloud drawing method
 */
void star::visualize::DrawColoredPointCloud(const PointCloud3fRGB_Pointer &point_cloud)
{
    std::string window_title = "3D Viewer";
    // boost::shared_ptr <pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer(window_title));
    // viewer->setBackgroundColor(0, 0, 0);
    // pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(point_cloud);
    // viewer->addPointCloud<pcl::PointXYZRGB>(point_cloud, rgb, "sample cloud");
    // viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "sample cloud");
    // while (!viewer->wasStopped()) {
    //     viewer->spinOnce(100);
    // }
}

void star::visualize::SaveColoredPointCloud(const PointCloud3fRGB_Pointer &point_cloud,
                                            const std::string &path)
{
    // redefine pcd size
    point_cloud->width = 1;
    point_cloud->height = point_cloud->points.size();
    pcl::io::savePCDFileASCII(path, *point_cloud);
}

void star::visualize::DrawColoredPointCloud(
    const star::GArray<float4> &vertex,
    const star::GArray<float4> &color_time)
{
    auto point_cloud = downloadColoredPointCloud(vertex, color_time);
    DrawColoredPointCloud(point_cloud);
}

void star::visualize::DrawColoredPointCloud(
    const star::GArrayView<float4> &vertex,
    const star::GArrayView<float4> &color_time)
{
    GArray<float4> vertex_array((float4 *)vertex.Ptr(), vertex.Size());
    GArray<float4> color_time_array((float4 *)color_time.Ptr(), color_time.Size());
    DrawColoredPointCloud(vertex_array, color_time_array);
}

void star::visualize::DrawColoredPointCloud(cudaTextureObject_t vertex_map, cudaTextureObject_t color_time_map)
{
    auto cloud = downloadColoredPointCloud(vertex_map, color_time_map, true);
    DrawColoredPointCloud(cloud);
}

void star::visualize::SaveColoredPointCloud(
    cudaTextureObject_t vertex_map,
    cudaTextureObject_t color_time_map,
    const std::string &path)
{
    auto cloud = downloadColoredPointCloud(vertex_map, color_time_map, true);
    SaveColoredPointCloud(cloud, path);
}

void star::visualize::SaveColoredPointCloud(
    const GArrayView<float4> &vertex,
    const GArrayView<float4> &color_time,
    const std::string &path)
{
    GArray<float4> vertex_array((float4 *)vertex.Ptr(), vertex.Size());
    GArray<float4> color_time_array((float4 *)color_time.Ptr(), color_time.Size());
    auto cloud = downloadColoredPointCloud(vertex_array, color_time_array, true);
    SaveColoredPointCloud(cloud, path);
}

void star::visualize::SaveColoredPointCloudWithNormal(
    const GArrayView<float4> &vertex,
    const GArrayView<float4> &color_time,
    const GArrayView<float4> &normal,
    const std::string &path)
{
    GArray<float4> vertex_array((float4 *)vertex.Ptr(), vertex.Size());
    GArray<float4> color_time_array((float4 *)color_time.Ptr(), color_time.Size());
    GArray<float4> normal_array((float4 *)normal.Ptr(), normal.Size());

    auto color_cloud = downloadColoredPointCloud(vertex_array, color_time_array, true);
    const auto normal_cloud = downloadNormalCloud(normal_array);

    PointRGBNormalCloud_Pointer point_rgb_normal_cloud = boost::make_shared<PointRGBNormalCloud>();
    pcl::concatenateFields(*color_cloud, *normal_cloud, *point_rgb_normal_cloud);

    // redefine pcd size
    point_rgb_normal_cloud->width = 1;
    point_rgb_normal_cloud->height = point_rgb_normal_cloud->points.size();
    pcl::io::savePCDFileASCII(path, *point_rgb_normal_cloud);
}

void star::visualize::SaveHeatedPointCloud(
    cudaTextureObject_t vertex_map,
    cudaTextureObject_t heat_map,
    const std::string &path,
    const float scale)
{
    auto cloud = downloadHeatedPointCloud(vertex_map, heat_map, scale);
    SaveColoredPointCloud(cloud, path);
}

void star::visualize::SaveHeatedPointCloud(
    const GArrayView<float4> &vertex_array,
    const GArrayView<unsigned> &heat_array,
    const std::string &path,
    const float scale)
{
    auto cloud = downloadHeatedPointCloud(vertex_array, heat_array, scale);
    SaveColoredPointCloud(cloud, path);
}

void star::visualize::SaveHeatedPointCloud(
    const GArrayView<float4> &vertex_array,
    const GArrayView<float> &heat_array,
    const std::string &path,
    const float scale)
{
    auto cloud = downloadHeatedPointCloud(vertex_array, heat_array, scale);
    SaveColoredPointCloud(cloud, path);
}

void star::visualize::SaveHeatedPointCloudWithNormal(
    const GArrayView<float4> &vertex_array,
    const GArrayView<float4> &normal_array,
    const GArrayView<unsigned> &heat_array,
    const std::string &path,
    const float scale,
    const std::string &color_map)
{
    auto color_cloud = downloadHeatedPointCloud(vertex_array, heat_array, scale, color_map);

    GArray<float4> device_normal_array((float4 *)normal_array.Ptr(), normal_array.Size());
    const auto normal_cloud = downloadNormalCloud(device_normal_array);
    PointRGBNormalCloud_Pointer point_rgb_normal_cloud = boost::make_shared<PointRGBNormalCloud>();
    pcl::concatenateFields(*color_cloud, *normal_cloud, *point_rgb_normal_cloud);

    // redefine pcd size
    point_rgb_normal_cloud->width = 1;
    point_rgb_normal_cloud->height = point_rgb_normal_cloud->points.size();
    pcl::io::savePCDFileASCII(path, *point_rgb_normal_cloud);
}

void star::visualize::SaveConfidencePointCloud(
    const GArrayView<float4> &vertex_confid,
    const std::string &path,
    const float scale)
{
    auto cloud = downloadConfidencePointCloud(vertex_confid, scale);
    SaveColoredPointCloud(cloud, path);
}

void star::visualize::SaveConfidencePointCloud(
    cudaTextureObject_t vertex_confid_map,
    const std::string &path,
    const float scale)
{
    auto cloud = downloadConfidencePointCloud(vertex_confid_map, scale);
    SaveColoredPointCloud(cloud, path);
}

void star::visualize::SaveTimePointCloud(
    const GArrayView<float4> &vertex,
    const GArrayView<float4> &color_time,
    const std::string &path,
    const float current_time)
{
    auto cloud = downloadTimePointCloud(vertex, color_time, current_time);
    SaveColoredPointCloud(cloud, path);
}

/* The method to draw matched cloud pair
 */
void star::visualize::DrawMatchedCloudPair(
    const PointCloud3f_Pointer &cloud_1,
    const PointCloud3f_Pointer &cloud_2)
{
    std::string window_title = "Matched Viewer";
    // boost::shared_ptr <pcl::visualization::PCLVisualizer> viewer(
    //         new pcl::visualization::PCLVisualizer(window_title));
    // viewer->setBackgroundColor(0, 0, 0);
    // pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> handler_1(cloud_1, 255, 0, 0);
    // viewer->addPointCloud<pcl::PointXYZ>(cloud_1, handler_1, "cloud 1");
    // pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> handler_2(cloud_2, 0, 255, 255);
    // viewer->addPointCloud<pcl::PointXYZ>(cloud_2, handler_2, "cloud 2");
    // while (!viewer->wasStopped()) {
    //     viewer->spinOnce(100);
    // }
}

void star::visualize::DrawMatchedCloudPair(
    const PointCloud3f_Pointer &cloud_1,
    const PointCloud3f_Pointer &cloud_2,
    const Eigen::Matrix4f &from1To2)
{
    PointCloud3f_Pointer transformed_cloud_1 = transformPointCloud(cloud_1, from1To2);
    DrawMatchedCloudPair(transformed_cloud_1, cloud_2);
}

void star::visualize::DrawMatchedCloudPair(
    cudaTextureObject_t cloud_1,
    const star::GArray<float4> &cloud_2,
    const star::Matrix4f &from1To2)
{
    const auto h_cloud_1 = downloadPointCloud(cloud_1);
    const auto h_cloud_2 = downloadPointCloud(cloud_2);
    DrawMatchedCloudPair(h_cloud_1, h_cloud_2, from1To2);
}

void star::visualize::DrawMatchedCloudPair(
    cudaTextureObject_t cloud_1,
    const GArrayView<float4> &cloud_2,
    const Matrix4f &from1To2)
{
    DrawMatchedCloudPair(
        cloud_1,
        GArray<float4>((float4 *)cloud_2.Ptr(), cloud_2.Size()),
        from1To2);
}

void star::visualize::DrawMatchedCloudPair(
    cudaTextureObject_t cloud_1,
    cudaTextureObject_t cloud_2,
    const star::Matrix4f &from1To2)
{
    const auto h_cloud_1 = downloadPointCloud(cloud_1);
    const auto h_cloud_2 = downloadPointCloud(cloud_2);
    DrawMatchedCloudPair(h_cloud_1, h_cloud_2, from1To2);
}

void star::visualize::SaveMatchedCloudPair(
    const PointCloud3f_Pointer &cloud_1,
    const PointCloud3f_Pointer &cloud_2,
    const std::string &cloud_1_name, const std::string &cloud_2_name)
{
    auto color_cloud_1 = addColorToPointCloud(cloud_1, make_uchar4(245, 0, 0, 255));
    auto color_cloud_2 = addColorToPointCloud(cloud_2, make_uchar4(200, 200, 200, 255));
    SaveColoredPointCloud(color_cloud_1, cloud_1_name);
    SaveColoredPointCloud(color_cloud_2, cloud_2_name);
}

void star::visualize::SaveMatchedCloudPair(
    const PointCloud3f_Pointer &cloud_1,
    const PointCloud3f_Pointer &cloud_2,
    const Eigen::Matrix4f &from1To2,
    const std::string &cloud_1_name, const std::string &cloud_2_name)
{
    PointCloud3f_Pointer transformed_cloud_1 = transformPointCloud(cloud_1, from1To2);
    SaveMatchedCloudPair(transformed_cloud_1, cloud_2, cloud_1_name, cloud_2_name);
}

void star::visualize::SaveMatchedCloudPair(
    cudaTextureObject_t cloud_1,
    const GArray<float4> &cloud_2,
    const Eigen::Matrix4f &from1To2,
    const std::string &cloud_1_name, const std::string &cloud_2_name)
{
    const auto h_cloud_1 = downloadPointCloud(cloud_1);
    const auto h_cloud_2 = downloadPointCloud(cloud_2);
    SaveMatchedCloudPair(
        h_cloud_1,
        h_cloud_2,
        from1To2,
        cloud_1_name, cloud_2_name);
}

void star::visualize::SaveMatchedCloudPair(
    cudaTextureObject_t cloud_1,
    const GArrayView<float4> &cloud_2,
    const Eigen::Matrix4f &from1To2,
    const std::string &cloud_1_name, const std::string &cloud_2_name)
{
    SaveMatchedCloudPair(
        cloud_1,
        GArray<float4>((float4 *)cloud_2.Ptr(), cloud_2.Size()),
        from1To2,
        cloud_1_name, cloud_2_name);
}

/* The method to draw mached color point cloud
 */
void star::visualize::DrawMatchedRGBCloudPair(const PointCloud3fRGB_Pointer &cloud_1,
                                              const PointCloud3fRGB_Pointer &cloud_2)
{
    std::string window_title = "3D Viewer";
    // boost::shared_ptr <pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer(window_title));
    // viewer->setBackgroundColor(0, 0, 0);
    // pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> handler_1(cloud_1);
    // viewer->addPointCloud<pcl::PointXYZRGB>(cloud_1, handler_1, "cloud_1");
    // pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> handler_2(cloud_2);
    // viewer->addPointCloud<pcl::PointXYZRGB>(cloud_2, handler_2, "cloud_2");

    // viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "cloud_1");
    // while (!viewer->wasStopped()) {
    //     viewer->spinOnce(100);
    // }
}

void star::visualize::DrawMatchedRGBCloudPair(
    const PointCloud3fRGB_Pointer &cloud_1,
    const PointCloud3fRGB_Pointer &cloud_2,
    const Eigen::Matrix4f &from1To2)
{
    PointCloud3fRGB_Pointer transformed_cloud_1 = transformPointCloudRGB(cloud_1, from1To2);

    // Hand in to drawer
    DrawMatchedRGBCloudPair(transformed_cloud_1, cloud_2);
}

void star::visualize::DrawMatchedCloudPair(
    cudaTextureObject_t vertex_map, cudaTextureObject_t color_time_map,
    const GArrayView<float4> &surfel_array,
    const GArrayView<float4> &color_time_array,
    const Eigen::Matrix4f &camera2world)
{
    auto cloud_1 = downloadColoredPointCloud(vertex_map, color_time_map, true);
    auto cloud_2 = downloadColoredPointCloud(
        GArray<float4>((float4 *)surfel_array.Ptr(), surfel_array.Size()),
        GArray<float4>((float4 *)color_time_array.Ptr(), color_time_array.Size()));
    DrawMatchedRGBCloudPair(cloud_1, cloud_2, camera2world);
}

void star::visualize::SaveMatchedPointCloud(
    const GArrayView<float4> &src_vertex_confid,
    const GArrayView<float4> &tar_vertex_confid,
    const std::string &path)
{
    STAR_CHECK_EQ(src_vertex_confid.Size(), tar_vertex_confid.Size());

    GArray<float4> src_vertex_array((float4 *)src_vertex_confid.Ptr(), src_vertex_confid.Size());
    const auto src_point_cloud = downloadPointCloud(src_vertex_array);
    GArray<float4> tar_vertex_array((float4 *)tar_vertex_confid.Ptr(), tar_vertex_confid.Size());
    const auto tar_point_cloud = downloadPointCloud(tar_vertex_array);

    // Build a normal point cloud from src & tar
    PointNormalCloud3f match_point_cloud; // From src pointing to tar
    for (auto i = 0; i < src_point_cloud->points.size(); ++i)
    {
        pcl::PointNormal p;
        p.x = src_point_cloud->points[i].x;
        p.y = src_point_cloud->points[i].y;
        p.z = src_point_cloud->points[i].z;

        p.normal_x = tar_point_cloud->points[i].x - src_point_cloud->points[i].x;
        p.normal_y = tar_point_cloud->points[i].y - src_point_cloud->points[i].y;
        p.normal_z = tar_point_cloud->points[i].z - src_point_cloud->points[i].z;

        match_point_cloud.points.push_back(p);
    }
    match_point_cloud.resize(src_point_cloud->points.size());

    pcl::io::savePCDFileASCII(path, match_point_cloud);
}

/* The method to draw fused surfel cloud
 */
void star::visualize::DrawFusedSurfelCloud(
    GArrayView<float4> surfel_vertex,
    GArrayView<unsigned> fused_indicator)
{
    STAR_CHECK_EQ(surfel_vertex.Size(), fused_indicator.Size());

    // Construct the host cloud
    PointCloud3f_Pointer fused_cloud(new PointCloud3f);
    PointCloud3f_Pointer unfused_cloud(new PointCloud3f);

    // Download it
    separateDownloadPointCloud(surfel_vertex, fused_indicator, fused_cloud, unfused_cloud);

    // Ok draw it
    DrawMatchedCloudPair(fused_cloud, unfused_cloud, Eigen::Matrix4f::Identity());
}

void star::visualize::DrawFusedSurfelCloud(
    star::GArrayView<float4> surfel_vertex,
    unsigned num_remaining_surfels)
{
    STAR_CHECK(surfel_vertex.Size() >= num_remaining_surfels);

    // Construct the host cloud
    PointCloud3f_Pointer remaining_cloud(new PointCloud3f);
    PointCloud3f_Pointer appended_cloud(new PointCloud3f);

    // Download it
    separateDownloadPointCloud(surfel_vertex, num_remaining_surfels, remaining_cloud, appended_cloud);

    // Ok draw it
    DrawMatchedCloudPair(remaining_cloud, appended_cloud);
}

void star::visualize::DrawFusedAppendedSurfelCloud(
    star::GArrayView<float4> surfel_vertex,
    star::GArrayView<unsigned int> fused_indicator,
    cudaTextureObject_t depth_vertex_map,
    star::GArrayView<unsigned int> append_indicator,
    const star::Matrix4f &world2camera)
{
    STAR_CHECK_EQ(surfel_vertex.Size(), fused_indicator.Size());

    // Construct the host cloud
    PointCloud3f_Pointer fused_cloud(new PointCloud3f);
    PointCloud3f_Pointer unfused_cloud(new PointCloud3f);

    // Download it
    separateDownloadPointCloud(surfel_vertex, fused_indicator, fused_cloud, unfused_cloud);
    auto h_append_surfels = downloadPointCloud(depth_vertex_map, append_indicator);

    // Draw it
    DrawMatchedCloudPair(fused_cloud, h_append_surfels, world2camera);
}

void star::visualize::DrawAppendedSurfelCloud(
    GArrayView<float4> surfel_vertex,
    cudaTextureObject_t depth_vertex_map,
    GArrayView<unsigned int> append_indicator,
    const star::Matrix4f &world2camera)
{
    auto h_surfels = downloadPointCloud(GArray<float4>((float4 *)surfel_vertex.Ptr(), surfel_vertex.Size()));
    auto h_append_surfels = downloadPointCloud(depth_vertex_map, append_indicator);
    DrawMatchedCloudPair(h_surfels, h_append_surfels, world2camera);
}

void star::visualize::DrawAppendedSurfelCloud(
    GArrayView<float4> surfel_vertex,
    cudaTextureObject_t depth_vertex_map,
    GArrayView<ushort2> append_pixel,
    const star::Matrix4f &world2camera)
{
    auto h_surfels = downloadPointCloud(GArray<float4>((float4 *)surfel_vertex.Ptr(), surfel_vertex.Size()));
    auto h_append_surfels = downloadPointCloud(depth_vertex_map, append_pixel);
    DrawMatchedCloudPair(h_surfels, h_append_surfels, world2camera);
}