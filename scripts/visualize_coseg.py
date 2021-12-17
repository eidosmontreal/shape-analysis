import datasets
import utils
import torch
import argparse
from os.path import join

parser = argparse.ArgumentParser()
parser.add_argument("--test", action='store_false')
parser.add_argument("--root", type=str, default='COSEG')
parser.add_argument("--cat", type=str, default='chairs')
parser.add_argument("--shuffle", action='store_true')
parser.add_argument("--n-samples", type=int, default=-1)
args = parser.parse_args()

ds = getattr(datasets,'COSEG')(join('data',args.root),classes=[args.cat],train=args.test)
loader = torch.utils.data.DataLoader(ds,shuffle=args.shuffle)

if args.n_samples == -1:
    args.n_samples = len(ds)
label_max = 4 if args.cat == 'chairs' else 3
for i,data in enumerate(loader):
    (v,e,f) = data[0]
    v = v[0]
    e = e[0]
    f = f[0]
    gt = data[1][0]
    v = (v-v.min())/(v.max()-v.min())
    utils.mesh2ply(v.numpy(),f.numpy(),gt.numpy()/label_max,fname='junk/{}{}.ply'.format('COSEG',i))
    if i == (args.n_samples+1):
        break


