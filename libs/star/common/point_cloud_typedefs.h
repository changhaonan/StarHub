#pragma once
#define WITH_PCL
#ifdef WITH_PCL
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
typedef pcl::PointCloud<pcl::PointXYZ> PointCloud3f;
typedef pcl::PointCloud<pcl::Normal> PointCloudNormal;
typedef pcl::PointCloud<pcl::PointNormal> PointNormalCloud3f;
typedef pcl::PointCloud<pcl::PointXYZRGB> PointCloud3fRGB;
typedef pcl::PointCloud<pcl::PointXYZRGBNormal> PointRGBNormalCloud;
typedef pcl::PointCloud<pcl::PointXYZ>::Ptr PointCloud3f_Pointer;
typedef pcl::PointCloud<pcl::Normal>::Ptr PointCloudNormal_Pointer;
typedef pcl::PointCloud<pcl::PointXYZRGB>::Ptr PointCloud3fRGB_Pointer;
typedef pcl::PointCloud<pcl::PointNormal>::Ptr PointNormalCloud3f_Pointer;
typedef pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr PointRGBNormalCloud_Pointer;
#endif
