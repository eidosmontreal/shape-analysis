import datasets
import utils
import open3d as o3d
import pdb
import torch
import argparse
from os.path import join

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default='HumanSegmentation')
parser.add_argument("--test", action='store_false')
parser.add_argument("--shuffle", action='store_true')
parser.add_argument("--n-samples", type=int, default=-1)
args = parser.parse_args()

ds = getattr(datasets,'HumanSegmentation')(join('data',args.dataset),train=args.test)
loader = torch.utils.data.DataLoader(ds,shuffle=args.shuffle)
(v,e,f),gt = ds[0]
vmax = v.max()
vmin = v.min()

if args.n_samples == -1:
    args.n_samples = len(ds)
for i,data in enumerate(loader):
    (v,e,f) = data[0]
    v = v[0]
    e = e[0]
    f = f[0]
    gt = data[1][0]
    v = (v-v.min())/(v.max()-v.min())
    utils.mesh2ply(v.numpy(),f.numpy(),gt.numpy()/7.0,fname='junk/{}_{}.ply'.format(args.dataset,i))
    if i == (args.n_samples+1):
        break


