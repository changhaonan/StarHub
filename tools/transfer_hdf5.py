""" Save hdf5 file to images
"""

import os
import json
import h5py
import numpy as np
import cv2
import easy3d_viewer


def transfer_hdf5(hdf5_file_path, output_dir, img_idx):
    # load file
    with h5py.File(hdf5_file_path, "r") as data:
        for key, value in data.items():
            if key == "colors":
                image = value
                image = np.array(image)
                image = image.astype(np.uint8)
                cv2.imwrite(
                    os.path.join(output_dir, f"frame_{img_idx:>06d}.color.png"), image
                )
            elif key == "depth":
                image = value
                image = np.array(image)
                image = (image * 1000.0).astype(np.uint16)  # convert to mm
                cv2.imwrite(
                    os.path.join(output_dir, f"frame_{img_idx:>06d}.depth.png"), image
                )
            elif key == "instance_segmaps":
                image = value
                image = np.array(image)
                cv2.imwrite(
                    os.path.join(output_dir, f"frame_{img_idx:>06d}.seg.png"), image
                )


if __name__ == "__main__":
    output_root = "/home/robot-learning/Projects/StarHub/data/home1"
    # Transfer image data
    for img_idx in range(50):
        hdf5_file_path = (
            f"/home/robot-learning/Projects/StarHub/data/sim/sim1/{img_idx}.hdf5"
        )
        output_dir = "/home/robot-learning/Projects/StarHub/data/home1/cam-00"
        transfer_hdf5(hdf5_file_path, output_dir, img_idx)
    
    # Save context
    context = easy3d_viewer.Context()
    context.addCoord("origin")
    context.addCamera("cam-00", "cam-00", np.eye(4), np.array([600.0, 600.0, 256.0, 256.0]), 512, 512, clip_near=0.1, clip_far=10.0)
    context.saveContext(os.path.join(output_root, "context.json"))