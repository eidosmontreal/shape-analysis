import os
import sys
from os.path import join
import subprocess

import torch
import torch_geometric
import trimesh
import numpy as np

from psbody.mesh import Mesh
import meshutils
import utils

import tqdm

root = join('..','data','COSEG1000')

if len(sys.argv) > 1:
    target_num_vertices = int(sys.argv[1])
    for mode in ['train','test']:
        for cat in ['tele_aliens','chairs','vases']:
            file_loop = tqdm.tqdm(os.listdir(join(root,mode,cat,'shapes'))) 
            for mesh_path in file_loop:
                file_loop.set_description(mesh_path)

                mesh_full_path = join(root,mode,cat,'shapes',mesh_path)
                
                # Using TorchGeometric
                data = torch_geometric.io.read_off(mesh_full_path)
                mesh = Mesh()
                mesh.v = np.array(data.pos)
                mesh.f = np.array(data.face.t()).astype(int)
                
                # Open mesh using psbody and then downsample
                #mesh = Mesh(filename=mesh_full_path)
                factor = len(mesh.v) / float(target_num_vertices)
                if factor <= 1:
                    # If number of vertices in mesh are already less than target number, skip.
                    continue
                subprocess.call(['rm {}'.format(mesh_full_path)],shell=True)
                try:
                    M,A,D,U = meshutils.generate_transform_matrices(mesh,[factor])
                except:
                    print('Could not downsample {}'.format(mesh_path))
                    continue
                new_data = torch_geometric.data.Data()
                new_data.pos = torch.Tensor(M[-1].v)
                new_data.face = torch.from_numpy(M[-1].f.astype(float)).type(torch.LongTensor).t()
                torch_geometric.io.write_off(new_data,mesh_full_path)
                
                gt_path = join(root,mode,cat,'vert_face',mesh_path.split('.')[0] + '.txt')
                gt = np.loadtxt(gt_path)
                new_gt = D[0].dot(gt)
                with open(gt_path,'w') as gt_file:
                    for i in range(len(new_gt)):
                        print(int(new_gt[i]),file=gt_file)
