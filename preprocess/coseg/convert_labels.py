import argparse
import os
from os.path import join

import numpy as np
import torch
import trimesh
import yaml

config = yaml.safe_load(open("config.yml"))


def convert_labels(data_dir: str, new_gt_folder: str = "vert_gt") -> None:
    """
    Transforms face ground truth labels to vertex labels by assigning to each vertex the label that occurs the most among all incident faces.

    :param data_dir: Directory where data is stored
    :param new_gt_folder: Name of folder where new data will be stored:
    """
    for mode in ["train", "test"]:
        for cat in config["classes"]["coseg"]:
            files = os.listdir(join(data_dir, mode, cat, "shapes"))
            os.makedirs(join(data_dir, mode, cat, new_gt_folder))
            for f in files:
                data = trimesh.load(join(data_dir, mode, cat, "shapes", f))

                faces = torch.tensor(data.faces).view(-1)

                gt_fp = f.split(".")[0] + ".seg"
                gt = torch.tensor(np.loadtxt(join(data_dir, mode, cat, "gt", gt_fp)))
                gt = gt.view(len(gt), 1)
                gt = torch.cat((gt, gt, gt), dim=1).view(-1)

                new_gt = torch.zeros(len(data.vertices))

                with open(join(data_dir, mode, cat, new_gt_folder, gt_fp), "w") as f:
                    # For each vertex select label to be label which appears most frequently among all incident faces
                    for i in range(len(data.vertices)):
                        labels_of_neighbours = gt[faces == i]

                        tmp = np.zeros(int(gt.max()) + 1)
                        for label in labels_of_neighbours:
                            tmp[int(label)] += 1
                        new_gt[i] = tmp.argmax()
                        print(int(new_gt[i]), file=f)  # Store new label in new ground truth file


if __name__ == "__main__":
    """
    Run ``convert_labels`` with input ``data_dir``.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True, help="Directory where data is stored")
    args = parser.parse_args()
    convert_labels(args.data_dir)
