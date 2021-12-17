import datasets
import utils
import open3d as o3d
import pdb
import torch
import argparse
from os.path import join

import torch_geometric
import torch_geometric.datasets as datasets


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default='ShapeNet')
parser.add_argument("--train", action='store_true')
parser.add_argument("--shuffle", action='store_true')
parser.add_argument("--n-samples", type=int, default=20)
args = parser.parse_args()

if args.dataset == 'ShapeNet':
    ds = getattr(datasets,'ShapeNet')(join('data','ShapeNet'),split='train' if args.train else 'test')
elif args.dataset == 'ModelNet':
    ds = getattr(datasets,'ModelNet')(join('data','ModelNet'),train=args.train)
loader = torch_geometric.data.DataLoader(ds,shuffle=args.shuffle)

if args.n_samples == -1:
    args.n_samples = len(ds)
for i,data in enumerate(loader):
    v = data.pos
    v = (v-v.min())/(v.max()-v.min())
    utils.pcl2ply(v.numpy(),fname='junk/{}_{}.ply'.format(args.dataset,i))
    if i == (args.n_samples+1):
        break


