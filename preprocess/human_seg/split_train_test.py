import os
import subprocess
from os.path import join
from typing import Type

import numpy as np
import torch
import trimesh


def face_to_node(num_vertices: int, labels: Type[np.array], faces: Type[np.array], new_text_file: str) -> None:
    """
    Converts face ground truth labels to vertex ground truth labels by taking the most frequently occuring label about each vertex.

    :param num_vertices: Number of vertices in mesh
    :param labels: Labels of vertices
    :param faces: Face indices
    :param new_text_file: File to store new ground truth labels
    """
    faces_flattened = torch.tensor(faces).view(-1)
    labels = torch.tensor(labels).view(len(labels), 1)
    labels_flattened = torch.cat((labels, labels, labels), dim=1).view(-1)
    with open(new_text_file, "w") as f:
        for i in range(num_vertices):
            labels_of_neighbours = labels_flattened[faces_flattened == i]
            tmp = np.zeros(int(labels.max()) + 1)
            for l in labels_of_neighbours:
                tmp[int(l)] += 1
            vert_label = tmp.argmax()
            print(int(vert_label), file=f)


def split_train_test(data_dir: str) -> None:
    """
    Splits HumanSegmentation data into training and testing set. The SHREC data is used as the test set.

    :param data_dir: Directory to find stored data
    """
    train_mesh_dir = join(data_dir, "train", "meshes")
    os.makedirs(train_mesh_dir)

    train_gt_dir = join(data_dir, "train", "gt")
    os.makedirs(train_gt_dir)

    test_mesh_dir = join(data_dir, "test", "meshes")
    os.makedirs(test_mesh_dir)

    test_gt_dir = join(data_dir, "test", "gt")
    os.makedirs(test_gt_dir)
    for mode in ["train", "test"]:
        if mode == "train":
            for dataset in os.listdir(join(data_dir, "meshes", mode)):
                for m in os.listdir(join(data_dir, "meshes", mode, dataset)):
                    if dataset != "MIT_animation":
                        old_mesh_file = join(data_dir, "meshes", mode, dataset, m)
                        new_mesh_file = join(train_mesh_dir, "{}_".format(dataset) + m)
                        subprocess.call(
                            ["cp {} {}".format(old_mesh_file, new_mesh_file)],
                            shell=True,
                        )

                        if dataset == "adobe":
                            txt = m.split(".off")[0] + ".txt"
                        elif dataset == "faust":
                            txt = "faust_corrected.txt"
                        elif dataset == "scape":
                            txt = "scape_corrected.txt"
                        old_text_file = join(data_dir, "segs", mode, dataset, txt)
                        new_text_file = join(
                            train_gt_dir,
                            "{}_".format(dataset) + m.split(".")[0] + ".txt",
                        )

                        mesh = trimesh.load(old_mesh_file)
                        num_vertices = len(mesh.vertices)
                        indices = np.loadtxt(old_text_file)
                        faces = mesh.faces

                        face_to_node(num_vertices, indices, faces, new_text_file)
                    else:
                        for obj in os.listdir(join(data_dir, "meshes", mode, dataset, m, "meshes")):
                            pose = m.split("_")[1]
                            old_mesh_file = join(data_dir, "meshes", mode, dataset, m, "meshes", obj)
                            new_mesh_file = join(train_mesh_dir, "MIT_animation_" + pose + "_" + obj)

                            subprocess.call(
                                ["cp {} {}".format(old_mesh_file, new_mesh_file)],
                                shell=True,
                            )

                            old_text_file = join(
                                data_dir,
                                "segs",
                                mode,
                                "mit",
                                "mit_" + pose + "_corrected.txt",
                            )
                            new_text_file = join(
                                train_gt_dir,
                                "MIT_animation_" + pose + "_" + obj.split(".")[0] + ".txt",
                            )

                            mesh = trimesh.load(old_mesh_file)
                            num_vertices = len(mesh.vertices)
                            indices = np.loadtxt(old_text_file)
                            faces = mesh.faces

                            face_to_node(num_vertices, indices, faces, new_text_file)
        else:
            for off in os.listdir(join(data_dir, "meshes", mode, "shrec")):

                i = int(off.split(".")[0] if off[:2] != "12" else "12")
                old_mesh_file = join(data_dir, "meshes", mode, "shrec", off)
                new_mesh_file = join(test_mesh_dir, "shrec_{}.off".format(i))

                subprocess.call(["cp {} {}".format(old_mesh_file, new_mesh_file)], shell=True)

                txt = "shrec_{}_full.txt".format(i)
                old_text_file = join(data_dir, "segs", mode, "shrec", txt)
                new_text_file = join(test_gt_dir, "shrec_{}.txt".format(i))

                mesh = trimesh.load(old_mesh_file)
                num_vertices = len(mesh.vertices)
                indices = np.loadtxt(old_text_file)
                faces = mesh.faces

                face_to_node(num_vertices, indices, faces, new_text_file)


if __name__ == "__main__":
    """
    Runs ``split_train_test`` with given ``data_dir``.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True, help="Directory where data is stored")
    args = parser.parse_args()

    split_train_test(args.data_dir)
