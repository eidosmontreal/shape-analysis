import os
from os.path import join
from typing import List, Tuple

import numpy as np
import torch
import torch_geometric
import trimesh
import yaml
from torch import FloatTensor, LongTensor, sparse
from torch_geometric.transforms import AddSelfLoops, FaceToEdge

config = yaml.safe_load(open("config.yml"))


class COSEG(torch.utils.data.Dataset):
    def __init__(self, root: str, classes: List[str] = None, train: bool = True, normalize: bool = True):
        """
        COSEG ``dataset``

        :param root: Folder path where data is stored
        :param classes: Classes of COSEG (tele-aliens, chairs, vases)
        :param train: Boolean indicating whether to load training or test data
        :param normalize: Boolean indicating to normalize vertices to unit box
        """
        if classes is None:
            classes = ["chairs"]
        else:
            diff = set(classes) - set(config["classes"]["coseg"])
            assert (
                len(diff) == 0
            ), f"{diff} are not available classes for COSEG, please choose among {config['classes']['coseg']}."
        mode = "train" if train else "test"
        files = []
        for c in classes:
            files_in_class = os.listdir(join(root, mode, c, "shapes"))
            files += [
                [
                    join(root, mode, c, "shapes", x),
                    join(root, mode, c, "vert_gt", x.split(".")[0] + ".seg"),
                ]
                for x in files_in_class
            ]

        self.root = root
        self.files = files
        self.face_to_edge = FaceToEdge(False)
        self.add_self_loops = AddSelfLoops()
        self.normalize = normalize

    def __len__(self) -> int:
        """
        Returns length of dataset.
        """
        return len(self.files)

    def __getitem__(self, idx: int) -> Tuple[Tuple[FloatTensor, LongTensor, LongTensor], LongTensor]:
        """
        Returns data and target of dataset at index ``idx``.
        """
        mesh_fp, gt = self.files[idx]

        # Load mesh data using trimesh
        mesh = trimesh.load(mesh_fp)
        vertices = torch.Tensor(mesh.vertices)
        faces = torch.Tensor(mesh.faces)

        # Convert to torch.geometric.data.Data type in order to use face_to_edge
        data = torch_geometric.data.Data(pos=vertices, face=faces.type(torch.long).t())
        data = self.add_self_loops(self.face_to_edge(data))
        edges = data.edge_index

        target = torch.Tensor(np.loadtxt(gt)).type(torch.LongTensor) - 1

        if vertices.shape[0] != int(faces.max()) + 1:
            # Sometimes vertices are unused in faces, this filters for only vertices laying on faces
            vertices = vertices[: faces.max() + 1]
            target = target[: faces.max() + 1]

        if self.normalize:
            vertices = (vertices - vertices.min()) / (vertices.max() - vertices.min())

        return (vertices, edges, faces), target


class HumanSegmentation(torch.utils.data.Dataset):
    def __init__(self, root: str, train: bool = True, normalize: bool = True):
        """
        HumanSegmentation ``dataset``

        :param root: Folder path where data is stored
        :param train: Boolean indicating whether to load training or test data
        :param normalize: Boolean indicating to normalize vertices to unit box
        """

        mode = "train" if train else "test"
        mesh_files = os.listdir(join(root, mode, "meshes"))
        files = [[join(root, mode, "meshes", x), join(root, mode, "gt", x.split(".")[0] + ".txt")] for x in mesh_files]

        self.root = root
        self.files = files
        self.face_to_edge = FaceToEdge(False)
        self.add_self_loops = AddSelfLoops()
        self.normalize = normalize

    def __len__(self) -> int:
        """
        Returns length of dataset.
        """
        return len(self.files)

    def __getitem__(self, idx: int) -> Tuple[Tuple[FloatTensor, LongTensor, LongTensor], LongTensor]:
        """
        Returns data and target of dataset at index ``idx``.
        """
        mesh_fp, gt = self.files[idx]

        # Load mesh data using trimesh
        mesh = trimesh.load(mesh_fp)
        vertices = torch.Tensor(mesh.vertices)
        faces = torch.Tensor(mesh.faces)

        # Convert to torch.geometric.data.Data type in order to use face_to_edge
        data = torch_geometric.data.Data(pos=vertices, face=faces.type(torch.long).t())
        data = self.add_self_loops(self.face_to_edge(data))
        edges = data.edge_index

        target = torch.Tensor(np.loadtxt(gt)).type(torch.LongTensor) - 1

        if self.normalize:
            vertices = (vertices - vertices.min()) / (vertices.max() - vertices.min())

        return (vertices, edges, faces), target


class MeshCNNDataset(torch.utils.data.Dataset):
    def __init__(self, root: str, train: bool = True, normalize: bool = True):
        """
        SHREC 16 ``dataset``

        :param root: Folder path where data is stored
        :param train: Boolean indicating whether to load training or test data
        :param normalize: Boolean indicating to normalize vertices to unit box
        """
        mode = "train" if train else "test"

        if "SHREC16" in root:
            self.class_dict = yaml.safe_load(open(join("preprocess", "shrec16.yml")))
        elif "CUBES" in root:
            self.class_dict = yaml.safe_load(open(join("preprocess", "cubes.yml")))

        files = []
        for c in self.class_dict.keys():
            c_idx = self.class_dict[c]
            files_in_class = os.listdir(join(root, c, mode))
            files += [[join(root, c, mode, x), c_idx] for x in files_in_class]

        self.root = root
        self.files = files
        self.face_to_edge = FaceToEdge(False)
        self.add_self_loops = AddSelfLoops()
        self.normalize = normalize

    def __len__(self) -> int:
        """
        Returns length of dataset.
        """
        return len(self.files)

    def __getitem__(self, idx: int) -> Tuple[Tuple[FloatTensor, LongTensor, LongTensor], int]:
        """
        Returns data and target of dataset at index ``idx``.
        """
        mesh_fp, target = self.files[idx]

        # Load mesh data using trimesh
        mesh = trimesh.load(mesh_fp)
        vertices = torch.Tensor(mesh.vertices)
        faces = torch.Tensor(mesh.faces)

        # Convert to torch.geometric.data.Data type in order to use face_to_edge
        data = torch_geometric.data.Data(pos=vertices, face=faces.type(torch.long).t())
        data = self.add_self_loops(self.face_to_edge(data))
        edges = data.edge_index

        if len(vertices) == 250:
            # If we're short 2 vertices just concatenate the last vertex twice
            vertices = torch.cat([vertices, vertices[-1].view(1, 3), vertices[-1].view(1, 3)], dim=0)

        if self.normalize:
            vertices = (vertices - vertices.min()) / (vertices.max() - vertices.min())

        return (vertices, edges, faces), target
