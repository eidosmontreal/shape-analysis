import torch
import torch.nn.functional as f

import models
import datasets
import utils

import argparse
import json
from os.path import join

device = 'cuda' if torch.cuda.is_available() else 'cpu'

parser = argparse.ArgumentParser()
parser.add_argument("--root", type=str, help="Root directory contains pretrained weights and experiments args")
parser.add_argument("--n-samples", type=int, default=3, help="Number of samples to demo on")
args = parser.parse_args()

model, train_args = utils.get_model(args.root,True)
model.eval()
model = model.to(device)

utils.print_dict(train_args)

data_root = join('data',train_args['dataset'])
if train_args['dataset'] == 'COSEG':
    dataset = getattr(datasets,'COSEG')(data_root,classes=train_args['classes'],train=False)
    target_max = 4.0
elif 'HumanSegmentation' in train_args['dataset']:
    dataset = getattr(datasets,'HumanSegmentation')(data_root,train=False)
    target_max = 8.0
dataloader = torch.utils.data.DataLoader(dataset,shuffle=False)

for i,data in enumerate(dataloader):
    (v,e,f),target = data
    v = v[0].to(device)
    e = e[0].to(device)
    f = f[0].to(device)
    target = target[0].to(device)
    
    if train_args['in_features'] == 1:
        x = torch.ones((len(v),1)).to(device)
    elif train_args['in_features'] == 3:
        x = v.clone()

    out = model(x,v,e,f)
    pred = torch.nn.functional.softmax(out,dim=1).max(1)[1]
    accuracy = pred.eq(target).sum().float()/len(target)
    print('%.3f accuracy for sample %d.'%(float(accuracy),i))
    
    v = (v-v.min())/(v.max()-v.min())
    utils.mesh2ply(v.cpu().numpy(),f.cpu().numpy(),weights=pred.cpu().numpy()/target_max,fname=join('junk','pred{}.ply'.format(i)))
    utils.mesh2ply(v.cpu().numpy(),f.cpu().numpy(),weights=target.cpu().numpy()/target_max,fname=join('junk','gt{}.ply'.format(i)))
    if i+1 == args.n_samples:
        break
