import torch
import trimesh

import models
import datasets

import json
from os.path import join
import tqdm

log_dir = '/home/babbasi/Perforce/babbasi_MTLWKS1562a_271/AI/ML/StyleTransfer/shape_analysis/logs/COSEG/segmentation/tele_aliens_face'
device ='cuda'
with open(join(log_dir,'args.json')) as tmp:
    args = json.load(tmp)

model = getattr(models,args['model'])(args['in_features'],args['out_features'],**args)
model = model.cuda()
model.eval()

aliens = datasets.COSEG('data/COSEG',classes=['tele_aliens'],train=True)
(v_ex,e_ex,f_ex),target = aliens[5]
v_ex = v_ex.to(device)
e_ex = e_ex.to(device)
f_ex = f_ex.to(device)
model(v_ex,v_ex,e_ex,f_ex)
exemplar_metric = model.metric_per_vertex

vases = datasets.COSEG('data/COSEG',classes=['vases'],train=True)
(v,e,f),target = vases[5]
v = torch.nn.Parameter(v.to(device))
e = e.to(device)
f = f.to(device)

lr = 1e-4
optimizer = torch.optim.Adam([v],lr=lr,betas=(0.9,0.99))

n_its = 100
optimizer_loop = tqdm.tqdm(range(n_its))
for i in optimizer_loop:

    model(v,v,e,f)
    metric = model.metric_per_vertex

    loss = (exemplar_metric[2] - metric[2]).pow(2).sum()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()


