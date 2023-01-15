""" Compute the error between gt pose and estimated pose.
"""
import json
import numpy as np
import os
from scipy.spatial.transform import Rotation as R


def read_gt(gt_file):
    with open(gt_file) as file:
        gt_data = [line.rstrip() for line in file]
    return gt_data


def get_gt(gt_data, frame_index):
    info = gt_data[frame_index].split(" ")
    gt_quat = np.zeros(4)
    gt_trans = np.zeros(3)

    for i in range(len(gt_quat)):
        gt_quat[i] = np.float32(info[i])

    for i in range(len(gt_trans)):
        gt_trans[i] = np.float32(info[i + 4])

    return gt_quat, gt_trans


def get_gt_matrix(gt_quat, gt_trans):
    r = R.from_quat(gt_quat)
    rota_matrix = r.as_matrix()

    gt_trans_matrix = np.zeros((4, 4))
    gt_trans_matrix[:3, :3] = rota_matrix
    gt_trans_matrix[-1, :3] = np.float32(0)
    gt_trans_matrix[:3, -1] = gt_trans.T
    gt_trans_matrix[-1, -1] = np.float32(1)

    return rota_matrix


def read_json(json_file):
    with open(json_file) as F:
        data = json.load(F)
    return data


def get_value(data):
    coordinate_list = data["average_dq"]["vis"]["coordinate"]
    trans_matrix = np.zeros((4, 4))
    for i in range(4):
        for j in range(4):
            index = i * 4 + j
            trans_matrix[j][i] = coordinate_list[index]
            j += 1
        i += 1

    rota_matrix = trans_matrix[:3, :3]
    trans = trans_matrix[:3, -1]
    r = R.from_matrix(rota_matrix)
    quat = r.as_quat()

    return rota_matrix, trans


def quaternion_rotation_matrix(Q):
    q0, q1, q2, q3 = Q[0], Q[1], Q[2], Q[3]

    r00 = 2 * (q0 * q0 + q1 * q1) - 1
    r01 = 2 * (q1 * q2 - q0 * q3)
    r02 = 2 * (q1 * q3 + q0 * q2)

    r10 = 2 * (q1 * q2 + q0 * q3)
    r11 = 2 * (q0 * q0 + q2 * q2) - 1
    r12 = 2 * (q2 * q3 - q0 * q1)

    r20 = 2 * (q1 * q3 - q0 * q2)
    r21 = 2 * (q2 * q3 + q0 * q1)
    r22 = 2 * (q0 * q0 + q3 * q3) - 1

    rot_matrix = np.array([[r00, r01, r02], [r10, r11, r12], [r20, r21, r22]])
    return rot_matrix


def compute_rotation_matrix_diff(A, B):
    A = np.matrix(A)
    B = np.matrix(B)
    return np.linalg.norm(np.dot(A, B.I), 2)


def compute_mse_scalar(gt_trans, trans):
    mse_trans = np.square(np.subtract(gt_trans, trans)).mean()
    return mse_trans


def compute_error(est_data_dir, gt_file_name, output_file_name):
    test_file_name = "context.json"
    gt_data = read_gt(gt_file_name)
    folder_list = sorted(os.listdir(est_data_dir))
    diff_list = []
    for folder in folder_list[:]:
        frame = folder.split("_")[1]
        frame_int = int(frame)
        if frame_int == 0:
            diff_list.append((np.float32(0), np.float32(0)))
        else:
            print("frame is: " + frame)
            test_file = os.path.join(os.path.join(est_data_dir, folder), test_file_name)
            data = read_json(test_file)
            rotation_matrix, trans = get_value(data)
            gt_quat, gt_trans = get_gt(gt_data, frame_int)
            gt_rotation_matrix = get_gt_matrix(gt_quat, gt_trans)
            rotation_diff = compute_rotation_matrix_diff(
                gt_rotation_matrix, rotation_matrix
            )
            mse_trans = compute_mse_scalar(gt_trans, trans)
            diff_list.append((rotation_diff, mse_trans))
    # mkdir if not exist
    output_dir = os.path.dirname(output_file_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
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
    est_data_dir = os.path.join(project_root, "external/Easy3DViewer/public/test_data", dataset)
    gt_file_name = os.path.join(project_root, "data", dataset, "pose_gt.txt")
    output_file_name = os.path.join(project_root, "eval", dataset, "mse.txt")
    compute_error(est_data_dir, gt_file_name, output_file_name)
