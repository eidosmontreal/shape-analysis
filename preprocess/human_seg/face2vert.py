import os
from os.path import join

import torch
import numpy as np
import trimesh

root = '.'
mesh_folder = 'shapes' 
gt_folder = 'face_gt'
new_gt_folder = 'vert_gt'

files = os.listdir(join(root,mesh_folder))
for f in files:
    data = trimesh.load(join(root,mesh_folder,f))
    faces = data.faces
    faces = faces.view(-1)

    gt_fp = f.split('.')[0]+'.txt'
    gt = torch.tensor(np.loadtxt(join(root,gt_folder,gt_fp)))
    gt = torch.cat((gt,gt,gt))

    new_gt = torch.zeros(len(data.pos))
    
    for i in range(len(data.pos)):
        labels_of_neighbours = gt[faces == i]

        tmp = np.zeros(int(gt.max())+1)
        for l in labels_of_neighbours:
            tmp[int(l)] += 1
        new_gt[i] = tmp.argmax()


    with open(join(root,new_gt_folder,gt_fp),'w') as f:
        for i in new_gt:
            print(int(i),file=f)

    
