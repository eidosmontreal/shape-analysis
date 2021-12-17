import torch
from torch import FloatTensor, LongTensor, sparse
from torchvision import transforms

import os
from os.path import join

import trimesh
import numpy as np
import torch_geometric
from torch_geometric import io 
from torch_geometric.transforms import FaceToEdge

from typing import List, Tuple

class COSEG(torch.utils.data.Dataset):
    def __init__(self,root: str,classes: List[str]=['chairs'],train: bool=True):
        """
        COSEG ``dataset``

        :param root: Folder path where data is stored
        :param classes: Classes of COSEG (tele-aliens, chairs, vases) 
        :param train: Boolean indicating whether to load training or test data
        """
        mode = 'train' if train else 'test'
        files = []
        for c in classes:
            assert c in ['tele_aliens','chairs','vases'], '{} is not a COSEG class'.format(c)
            files_in_class = os.listdir(join(root,mode,c,'shapes'))
            files += [[join(root,mode,c,'shapes',x),join(root,mode,c,'vert_face',x.split('.')[0]+'.txt')] for x in files_in_class]
        
        self.root = root
        self.files = files
        self.face_to_edge = FaceToEdge(False)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self,idx: int) -> Tuple[Tuple[FloatTensor,sparse.FloatTensor,LongTensor],LongTensor]:
        mesh_fp,gt = self.files[idx]
        data = self.face_to_edge(io.read_off(mesh_fp))
        
        vertices = data.pos
        edges = data.edge_index
        faces = data.face.t()
        
        target = torch.Tensor(np.loadtxt(gt)).type(torch.LongTensor)-1
        
        if vertices.shape[0] != faces.max():
           # Sometimes vertices are unused in faces, this filters for only vertices laying on faces 
           vertices = vertices[:faces.max()+1]
           target = target[:faces.max()+1]
        
        return (vertices,edges,faces), target 

class HumanSegmentation(torch.utils.data.Dataset):
    """
    HumanSegmentation ``dataset``

    :param root: Folder path where data is stored
    :param train: Boolean indicating whether to load training or test data
    """
    def __init__(self,root: str,train: bool=True):
        self.mode = 'train' if train else 'test'
        self.root = join(root,self.mode)
        self.files = os.listdir(join(self.root,'meshes'))

        self.face_to_edge = FaceToEdge(False)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self,idx: int) -> Tuple[Tuple[FloatTensor,sparse.FloatTensor,LongTensor],LongTensor]:
        mesh_file = self.files[idx]
        
        mesh = trimesh.load(join(self.root,'meshes',mesh_file))
        vertices = torch.Tensor(mesh.vertices)
        faces = torch.Tensor(mesh.faces)
        
        data = torch_geometric.data.Data(pos = vertices,face = faces.type(torch.long).t())
        data = self.face_to_edge(data)
        edges = data.edge_index
        
        txt_file = mesh_file.split('.')[0]+'.txt'
        txt_path = join(self.root,'gt',txt_file)
        target = torch.Tensor(np.loadtxt(txt_path)).type(torch.LongTensor)-1

        return (vertices,edges,faces), target 

