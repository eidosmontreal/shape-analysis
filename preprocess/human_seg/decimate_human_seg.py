import os
import sys
from os.path import join
import subprocess

import torch
import trimesh
import numpy as np

from psbody.mesh import Mesh
import meshutils
import utils

import tqdm

import pdb
root = join('HumanSegmentation_ds_1000')

if len(sys.argv) > 1:
    target_num_vertices = int(sys.argv[1])
    for mode in ['train','test']:
        file_loop = tqdm.tqdm(os.listdir(join(root,mode,'meshes'))) 
        for mesh_path in file_loop:
            file_loop.set_description(mesh_path)

            mesh_full_path = join(root,mode,'meshes',mesh_path)
            ext = mesh_path.split('.')[-1]
            
            # Rewrite all meshes in my .obj format
            mesh = trimesh.load(mesh_full_path)
            new_mesh_path = mesh_full_path.split(ext)[0]+'obj'
            subprocess.call(['rm {}'.format(mesh_full_path)],shell=True)
            utils.mesh2obj(mesh.vertices,mesh.faces,fname=new_mesh_path)
            mesh_full_path = new_mesh_path 

            # Open mesh using psbody and then downsample
            mesh = Mesh(filename=mesh_full_path)
            vertices = mesh.v
            subprocess.call(['rm {}'.format(mesh_full_path)],shell=True)
            factor = len(vertices) / float(target_num_vertices)
            if factor <= 1:
                # If number of vertices in mesh are already less than target number, skip.
                continue
            M,A,D,U = meshutils.generate_transform_matrices(mesh,[factor])
            new_vertices = M[-1].v
            new_faces = M[-1].f
            utils.mesh2obj(new_vertices,new_faces,fname=mesh_full_path.split('.')[0]+'.obj')
            
            gt_path = join(root,mode,'gt',mesh_path.split('.')[0] + '.txt')
            gt = np.loadtxt(gt_path)
            new_gt = D[0].dot(gt)
            with open(gt_path,'w') as gt_file:
                for i in range(len(new_gt)):
                    print(int(new_gt[i]),file=gt_file)
