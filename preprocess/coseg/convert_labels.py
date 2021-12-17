import os, subprocess, sys
from os.path import join

import torch
import numpy as np
import trimesh

"""
Transforms face ground truth labels to vertex labels by assigning to each vertex the label that occurs the most among all incident faces.
"""

#root = join('..','data','COSEG')
assert len(sys.argv) > 1, "Please specify data directory."
root = sys.argv[1]
new_gt_folder = 'vert_gt'


for mode in ['train','test']:
    for cat in ['tele_aliens','chairs','vases']:
        files = os.listdir(join(root,mode,cat,'shapes'))
        subprocess.call(['mkdir -p {}'.format(join(root,mode,cat,new_gt_folder))],shell=True)
        for f in files:
            data = trimesh.load(join(root,mode,cat,'shapes',f))
       
            faces = torch.tensor(data.faces).view(-1)
        
            gt_fp = f.split('.')[0]+'.seg'
            gt = torch.tensor(np.loadtxt(join(root,mode,cat,'gt',gt_fp)))
            gt = gt.view(len(gt),1)
            gt = torch.cat((gt,gt,gt),dim=1).view(-1)
        
            new_gt = torch.zeros(len(data.vertices))
            
            with open(join(root,mode,cat,new_gt_folder,gt_fp),'w') as f:
                # For each vertex select label to be label which appears most frequently among all incident faces
                for i in range(len(data.vertices)):
                    labels_of_neighbours = gt[faces == i]
        
                    tmp = np.zeros(int(gt.max())+1)
                    for l in labels_of_neighbours:
                        tmp[int(l)] += 1
                    new_gt[i] = tmp.argmax()
                    print(int(new_gt[i]),file=f) # Store new label in new ground truth file

    
