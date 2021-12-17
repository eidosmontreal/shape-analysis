import torch
from torch_geometric.io import read_obj
from torch_geometric.transforms import FaceToEdge
import geomloss

import models
import datasets
import utils

import json
from os.path import join
import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model_args = {'info':'face','n_hidden':100,'embedding_dim':2}
model = getattr(models,'MetricConvNet')(3,3,**model_args)
model = model.to(device)

vases = datasets.COSEG('data/COSEG',classes=['vases'],train=True)
(point_cloud,_,__),target = vases[5]
point_cloud = point_cloud.to(device)

#f2e = FaceToEdge(False)
#mesh = f2e(read_obj('data/misc/icosphere.obj'))
#v = mesh.pos.to(device)
#e = torch.sparse.FloatTensor(mesh.edge_index,torch.ones(mesh.edge_index.shape[1])).to(device)
#f = mesh.face.t().to(device)
(v,e,f),_ = vases[10]
v = v.to(device)
e = e.to(device)
f = f.to(device)


lr = 1e-4
optimizer = torch.optim.Adam(model.parameters(),lr=lr,betas=(0.9,0.99))

n_its = 1000
optimizer_loop = tqdm.tqdm(range(n_its))
sinkhorn_loss = geomloss.SamplesLoss(loss='sinkhorn').to(device)
for i in optimizer_loop:

    diff = model(v,v,e,f)
    v = v + diff
    loss = sinkhorn_loss(v,point_cloud) 
    v = v.detach() 
    
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

utils.mesh2obj(v.data.cpu().numpy(),f.data.cpu().numpy(),fname='junk/deformed_sphere.obj')

