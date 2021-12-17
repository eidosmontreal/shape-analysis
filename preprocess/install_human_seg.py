import os
from os.path import join
import subprocess

import torch
import trimesh
import numpy as np

from psbody.mesh import Mesh
import tmp

data_url = 'https://www.dropbox.com/sh/cnyccu3vtuhq1ii/AADgGIN6rKbvWzv0Sh-Kr417a?dl=0&preview=human_benchmark_sig_17.zip'
data_path = join('..','data','human_benchmark_sig_17.zip')
if not os.path.exists(data_path):
    raise RuntimeError('Please download {} from {} and move it to {}.'.format(
            'human_benchmark_sig_17.zip',
            data_url,
            join('..','data')))
subprocess.call(['unzip {} -d {}'.format(data_path,join('..','data'))],shell=True)
subprocess.call(['mv {} {}'.format(join('..','data','sig17_seg_benchmark'),join('..','data','HumanSegmentation'))],shell=True)

root = join('..','data','HumanSegmentation')
train_mesh_root = join('..','data','HumanSegmentation','train','meshes')
train_gt_root = join('..','data','HumanSegmentation','train','gt')

test_mesh_root = join('..','data','HumanSegmentation','test','meshes')
test_gt_root = join('..','data','HumanSegmentation','test','gt')

subprocess.call(['mkdir -p {} && mkdir -p {} && mkdir -p {} && mkdir -p {}'.format(train_mesh_root,train_gt_root,test_mesh_root,test_gt_root)],shell=True)

def face_to_node(num_vertices,indices,faces,new_text_file):
    vert_labels = torch.ones(num_vertices)
    for i,l in enumerate(indices):
        vert_labels[faces[i][0]] = l
    with open(new_text_file,'w') as f:
        for v in vert_labels:
            print(int(v),file=f)

for mode in ['train','test']:
    if mode == 'train':
        for dataset in os.listdir(join(root,'meshes',mode)):
            for m in os.listdir(join(root,'meshes',mode,dataset)):
                if dataset != 'MIT_animation':
                    old_mesh_file = join(root,'meshes',mode,dataset,m)
                    new_mesh_file = join(train_mesh_root,'{}_'.format(dataset)+m)
                    subprocess.call(['cp {} {}'.format(old_mesh_file,new_mesh_file)],shell=True)
                    
                    if dataset == 'adobe':
                        txt = m.split('.off')[0]+'.txt'
                    elif dataset == 'faust':
                        txt = 'faust_corrected.txt' 
                    elif dataset == 'scape':
                        txt = 'scape_corrected.txt'
                    old_text_file = join(root,'segs',mode,dataset,txt)
                    new_text_file = join(train_gt_root,'{}_'.format(dataset) + m.split('.')[0] + '.txt')

                    mesh = trimesh.load(old_mesh_file)
                    num_vertices = len(mesh.vertices)
                    indices = np.loadtxt(old_text_file)
                    faces = mesh.faces

                    face_to_node(num_vertices,indices,faces,new_text_file)
                else:
                    for obj in os.listdir(join(root,'meshes',mode,dataset,m,'meshes')):
                        pose = m.split('_')[1]
                        old_mesh_file = join(root,'meshes',mode,dataset,m,'meshes',obj)
                        new_mesh_file = join(train_mesh_root,'MIT_animation_' + pose +'_' + obj)
                        
                        subprocess.call(['cp {} {}'.format(old_mesh_file,new_mesh_file)],shell=True)
                        
                        old_text_file = join(root,'segs',mode,'mit','mit_' + pose + '_corrected.txt')
                        new_text_file = join(train_gt_root,'MIT_animation_' + pose + '_' + obj.split('.')[0]+'.txt')

                        mesh = trimesh.load(old_mesh_file)
                        num_vertices = len(mesh.vertices)
                        indices = np.loadtxt(old_text_file)
                        faces = mesh.faces

                        face_to_node(num_vertices,indices,faces,new_text_file)
    else:
        for off in os.listdir(join(root,'meshes',mode,'shrec')):
           
            i = int(off.split('.')[0] if off[:2] != '12' else '12')
            old_mesh_file = join(root,'meshes',mode,'shrec',off)
            new_mesh_file = join(test_mesh_root,'shrec_{}.off'.format(i))
            
            subprocess.call(['cp {} {}'.format(old_mesh_file,new_mesh_file)],shell=True)
            
            txt = 'shrec_{}_full.txt'.format(i)
            old_text_file = join(root,'segs',mode,'shrec',txt)
            new_text_file = join(test_gt_root,'shrec_{}.txt'.format(i))

            mesh = trimesh.load(old_mesh_file)
            num_vertices = len(mesh.vertices)
            indices = np.loadtxt(old_text_file)
            faces = mesh.faces


            face_to_node(num_vertices,indices,faces,new_text_file)
                    

if len(sys.argv) > 1:
    target_num_vertices = sys.argv[1]
    for mode in ['train','test']:
        for mesh_path in os.listdir(root,mode,'meshes')
            gt_path = join(root,mode,'gt',mesh_path.split('.')[0] + '.txt')
            mesh_path = join(root,mode,'meshes',mesh_path)
            if mesh_path.endswith('off'):
                mesh = trimesh.load(mesh_path)
                new_mesh_path = join(root,mode,'meshes',mesh_path.split('.')[0]+'.obj')
                utils.mesh2obj(mesh.vertices,mesh.faces,fname=new_mesh_path)
                subprocess.call(['rm {}'.format(mesh_path)],shell=True)
            
            gt = torch.Tensor(np.loadtxt(gt_path)).type(torch.LongTensor)

            
            mesh = Mesh(filename=new_mesh_path)
            factor = len(mesh.v) // target_num_vertices

            M,A,D,U = tmp.generate_transform_matrices(mesh,[factor])
            new_gt = torch.mm(D[0],gt)

