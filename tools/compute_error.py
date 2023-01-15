""" 
Compute the error between gt pose and estimated pose.
Note: ycb data consists of (x, y, z, rx, ry, rz, angle) for each frame.
"""
import json
import numpy as np
import os
from scipy.spatial.transform import Rotation as R


def read_gt(gt_file):
    with open(gt_file) as file:
        gt_data = [line.rstrip() for line in file]
    return gt_data


def get_gt_pose(gt_data, frame_index):
    info = gt_data[frame_index].split(" ")
    gt_axis_angle = np.zeros(4)
    gt_trans = np.zeros(3)

    for i in range(len(gt_trans)):
        gt_trans[i] = np.float32(info[i])

    for i in range(len(gt_axis_angle)):
        gt_axis_angle[i] = np.float32(info[i + 3])

    return gt_axis_angle, gt_trans


def get_rot_from_quat(gt_quat):
    r = R.from_quat(gt_quat)
    rota_matrix = r.as_matrix()
    return rota_matrix


def get_rot_from_axis_angle(gt_axis_angle):
    rot_vec = gt_axis_angle[:3] * gt_axis_angle[3]
    r = R.from_rotvec(rot_vec)
    rota_matrix = r.as_matrix()
    return rota_matrix


def read_json(json_file):
    with open(json_file) as F:
        data = json.load(F)
    return data


def get_context_pose(data, obj_name="average_dq"):
    coordinate_list = data[obj_name]["vis"]["coordinate"]
    trans_matrix = np.zeros((4, 4))
    for i in range(4):
        for j in range(4):
            index = i * 4 + j
            trans_matrix[j][i] = coordinate_list[index]
            j += 1
        i += 1

    rota_matrix = trans_matrix[:3, :3]
    trans = trans_matrix[:3, -1]
    return rota_matrix, trans


def compute_geodesic_dist(A, B):
    A = np.matrix(A)
    B = np.matrix(B)
    AB = np.dot(A, B.T)
    AB_axis_angle = R.from_matrix(AB).as_rotvec()
    return np.linalg.norm(AB_axis_angle)


def compute_mse_scalar(gt_trans, trans):
    mse_trans = np.square(np.subtract(gt_trans, trans)).mean()
    return mse_trans


def compute_error(
    est_data_dir,
    gt_file_name,
    output_file_name,
    obj_name="average_dq",
    est_frame_start=0,
):
    """
    Compute the error between gt pose and estimated pose.
    Note: the first frame is fixed to be identity for both gt and estimated pose.
    """
    test_file_name = "context.json"
    gt_data = read_gt(gt_file_name)
    folder_list = sorted(os.listdir(est_data_dir))
    diff_list = []

    # prepare the initial pose
    context_rot_m_init = np.identity(3)
    context_trans_init = np.zeros(3)
    gt_rot_m_init = np.identity(3)
    gt_trans_init = np.zeros(3)
    initialized = False

    # compute the error by frame
    for folder in folder_list[:]:
        context_file = os.path.join(os.path.join(est_data_dir, folder), test_file_name)
        context_data = read_json(context_file)
        real_frame_idx = int(context_data["extra_info"]["real_frame_idx"])
        if obj_name in context_data:
            if not initialized:
                gt_axis_angle, gt_trans_init = get_gt_pose(gt_data, real_frame_idx)
                gt_rot_m_init = get_rot_from_axis_angle(gt_axis_angle)
                context_rot_m_init, context_trans_init = get_context_pose(
                    context_data, obj_name
                )
                diff_list.append((np.float32(0), np.float32(0)))
                initialized = True
            else:
                context_rot_m, context_trans = get_context_pose(context_data, obj_name)
                gt_axis_angle, gt_trans = get_gt_pose(gt_data, real_frame_idx)
                gt_rot_m = get_rot_from_axis_angle(gt_axis_angle)
                # compute relative pose w.r.t. the initial pose
                context_rot_m = context_rot_m.dot(np.linalg.inv(context_rot_m_init))
                context_trans = context_trans - context_rot_m.dot(context_trans_init)
                gt_rot_m = gt_rot_m.dot(np.linalg.inv(gt_rot_m_init))
                gt_trans = gt_trans - gt_rot_m.dot(gt_trans_init)
                # compute diff
                mse_rot = compute_geodesic_dist(gt_rot_m, context_rot_m)
                mse_trans = compute_mse_scalar(gt_trans, context_trans)
                diff_list.append((mse_rot, mse_trans))
        else:
            continue
    # make summary
    diff_list = np.array(diff_list)
    average_diff = np.mean(diff_list, axis=0)
    print(f"MSE Rot: {average_diff[0]}, MSE Trans: {average_diff[1]}.")

    # mkdir if not exist
    output_dir = os.path.dirname(output_file_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # save mse to file
    with open(output_file_name, "w") as fp:
        [fp.write(str(item[0]) + " " + str(item[1]) + "\n") for item in diff_list]
        fp.close()


if __name__ == "__main__":
    # parse arguments
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="fastycb1")
    args = parser.parse_args()

    # generate path
    dataset = args.dataset
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    est_data_dir = os.path.join(
        project_root, "external/Easy3DViewer/public/test_data", dataset
    )
    gt_file_name = os.path.join(project_root, "data", dataset, "pose_gt.txt")
    output_file_name = os.path.join(project_root, "eval", dataset, "mse.txt")
    compute_error(est_data_dir, gt_file_name, output_file_name)
