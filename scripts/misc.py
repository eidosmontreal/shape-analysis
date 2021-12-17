import torch
import numpy as np

import datasets
import utils
import models

from os.path import join
import pdb, os, json
from tqdm import tqdm

import geomloss
import trimesh

root = join(os.path.dirname(os.path.realpath(__file__)),'..')
dataset = 'ShapeNetHKS_JSON'
dir_name = 'GCN20' 
log_dir = join(root,'logs',dataset,dir_name)

# Load dataset
if dataset == 'BasicShapesHKS':
    test_dataset = datasets.BasicShapesHKS(join(root,'data','BasicShapes'),time_steps=1)
else:
    test_dataset = getattr(datasets,dataset)(join(root,'data','ShapeNet','train'),classes=['chair'],time_steps=1)

# Load model with pretrained weights
with open(join(log_dir,'args.json')) as f:
    args = json.load(f)
args['save_features'] = True
model = getattr(models,args['model'])(args['in_features'],args['out_features'],**args)
model.load_state_dict(torch.load(join(log_dir,'weights.pth'))['weights'])
model.eval()
model = model.cuda()

to_numpy = lambda x: x.detach().cpu().numpy()
t = 0

wass_loss = geomloss.SamplesLoss(loss='sinkhorn',p=2,blur=0.05)
wass_loss = wass_loss.cuda()
#colors = cmap(np.linspace(0,1,num_its+1))

#Load target mesh.
ix = 2
hks, (V,E,F) = test_dataset[ix]
V = V.detach().cuda()
E = E.detach().cuda()
F = F.detach().cuda()
hks = hks[0].cuda()
target_pred,target_feats,_ = model(V,E)

#Load mesh to be deformed.
mesh = trimesh.load(join(root,'data','misc','low_poly_sphere.obj'))
vertices = torch.Tensor(mesh.vertices)
vertices = vertices.cuda()
vertices = (V.max(dim=0)[0]-V.min(dim=0)[0])*(vertices - vertices.min(dim=0)[0])/(vertices.max(dim=0)[0]-vertices.min(dim=0)[0]) + V.min(dim=0)[0]

faces = torch.Tensor(mesh.faces)
edges = utils.face_to_edge(faces).cuda()
edges_idx = edges._indices().cuda()

f_idx = 1 
lr=1e-2
min_lr = 5e-3 
def corr(X):
    Y = (X-X.mean(dim=0))/X.std(dim=0)
    return torch.mm(Y.t(),Y)/Y.shape[0]

k_subdivide = 300 
num_subdivide = 4
num_its = num_subdivide * k_subdivide 
scheduler = min_lr + (lr - min_lr)*(torch.cos(torch.linspace(0,np.pi,num_its))+1)/2
optimizer_loop = tqdm(range(num_its))
freq = 20
stage = 0
tfeats = target_feats[f_idx].detach()
for i in optimizer_loop:
    if i % k_subdivide == 0:
        if i > 0:
            # Subdivide mesh (each face is divided into 4 faces)
            utils.mesh2obj(to_numpy(vertices),faces,fname=join(root,'junk','transform_mesh{}_{}_{}.obj'.format(ix,f_idx,stage)))
            vertices = to_numpy(vertices)
            faces = to_numpy(faces).astype(int)

            mesh = trimesh.remesh.subdivide(vertices,faces)
            
            vertices = torch.Tensor(mesh[0]).cuda()
            faces = torch.Tensor(mesh[1])
            edges = utils.face_to_edge(faces).cuda()
        
        vertices = torch.nn.Parameter(vertices)
        optimizer = torch.optim.Adam([vertices],lr=scheduler[i],betas=(0.9,0.99),weight_decay=0)
        stage+=1

    optimizer.zero_grad()
    pred,feats,adj = model(vertices,edges)
    
    loss = 0
    #loss += 1e-1*(corr(feats[f_idx])-corr(target_feats[f_idx]).detach()).norm(p=2)
    loss += wass_loss(feats[f_idx],tfeats[torch.randperm(len(tfeats))[:len(vertices)],:])
    #loss += (vertices[edges_idx[0]] - vertices[edges_idx[1]]).norm()
    #loss += (adj.to_dense() - initial_attention_matrix.to_dense()).abs().sum()/adj._nnz() # Loss enforcing initial neighbourhoods

    loss.backward()
    optimizer.step()
    for param in optimizer.param_groups:
        param['lr'] = scheduler[i]
    optimizer_loop.set_description(str(loss.cpu().data.numpy()))

utils.mesh2ply(to_numpy(vertices),join(root,'junk','transform{}_{}.ply'.format(ix,f_idx)))
utils.mesh2ply(to_numpy(V),join(root,'junk','target{}_{}.ply'.format(ix,f_idx)))
utils.mesh2obj(to_numpy(vertices),faces,fname=join(root,'junk','transform_mesh{}_{}.obj'.format(ix,f_idx)))
utils.mesh2obj(to_numpy(V),to_numpy(F),fname=join(root,'junk','target_mesh{}_{}.obj'.format(ix,f_idx)))
