import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt


def point_cloud_denoise(point_cloud, alg="DBSCAN", enable_vis=False):
    if alg == "DBSCAN":
        # noise removal with DBSCAN using open3d
        labels = np.array(
            point_cloud.cluster_dbscan(eps=0.01, min_points=10, print_progress=True)
        )
        # only left the one with most labels
        max_label = max(labels, key=list(labels).count)
        print(
            "point cloud with {} points, remove {} points".format(
                len(labels), list(labels).count(-1)
            )
        )
        inlier_cloud = point_cloud.select_by_index(np.where(labels == max_label)[0])
        if enable_vis:
            outlier_cloud = point_cloud.select_by_index(np.where(labels != max_label)[0])
            outlier_cloud.paint_uniform_color([0, 1, 0])
            o3d.visualization.draw_geometries(
                [
                    inlier_cloud,
                    outlier_cloud,
                    o3d.geometry.TriangleMesh.create_coordinate_frame(),
                ]
            )
        return inlier_cloud
    elif alg == "BBOX":
        # filter out the points outside the bounding box
        bbox = o3d.geometry.OrientedBoundingBox(
            center=np.array([-0.025, 0.25, 0.3]),
            R=np.eye(3),
            extent=np.array([0.19, 0.08, 0.3]),
        )
        bbox.color = (0, 0, 1)
        inlier_cloud = point_cloud.crop(bbox)
        if enable_vis:
            o3d.visualization.draw_geometries(
                [inlier_cloud, bbox, o3d.geometry.TriangleMesh.create_coordinate_frame()]
            )
        # get the bounding box
        return point_cloud.crop(bbox)

    return point_cloud


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="", help="input point cloud")
    parser.add_argument("--output", type=str, default="", help="output point cloud")
    args = parser.parse_args()
    # read point cloud
    pcd = o3d.io.read_point_cloud(args.input)

    # denoise
    pcd_denoise = point_cloud_denoise(pcd, "BBOX")

    # apply possion reconstruction
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd_denoise, depth=9)
    o3d.visualization.draw_geometries([mesh])
