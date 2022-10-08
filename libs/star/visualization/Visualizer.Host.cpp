#include <pcl/visualization/pcl_visualizer.h>
#include <star/visualization/Visualizer.Host.h>

void star::visualize::DrawPointCloud(const PointCloud3f_Pointer &point_cloud)
{
    const std::string window_title = "simple point cloud viewer";
    pcl::visualization::PCLVisualizer viewer(window_title);
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> handler(point_cloud, 255, 255, 255);
    viewer.addPointCloud(point_cloud, "point cloud");
    viewer.addCoordinateSystem(2.0, "point cloud", 0);
    viewer.setBackgroundColor(0.05, 0.05, 0.05, 1);
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "point cloud");
    while (!viewer.wasStopped())
    {
        viewer.spinOnce();
    }
}