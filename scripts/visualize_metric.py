import models 
import datasets
import utils
import torch
import pdb
from os.path import join

import torch
import torch_geometric
import torch_geometric.transforms as T

device = 'cuda' if torch.cuda.is_available() else 'cpu'
idx = 5
to_numpy = lambda x : x.detach().cpu().numpy()
out_dir = 'junk'

#log_dir = '/home/babbasi/Perforce/babbasi_MTLWKS1562a_271/AI/ML/StyleTransfer/shape_analysis/logs/FAUST/correspondence/test_new_pen'
log_dir = '/home/babbasi/Perforce/babbasi_MTLWKS1562a_271/AI/ML/StyleTransfer/shape_analysis/logs/COSEG/segmentation/test_new_pen'
model, args = utils.get_model(log_dir,True) # Load model found in log_dir and load trained parameters

if 'HumanSegmentation' in args['dataset']:
    ds = getattr(datasets,'HumanSegmentation')(join('data',args['dataset']))
    (v,e,f),gt = ds[idx]
elif 'COSEG' in args['dataset']:
    ds = getattr(datasets,'COSEG')(join('data',args['dataset']),classes=args['classes'],train=False)
    (v,e,f),gt = ds[idx]
elif args['dataset'] == 'FAUST':
    ds = torch_geometric.datasets.FAUST(join('data','FAUST'),False,transform=T.FaceToEdge(False))
    data = ds[idx]
    gt = torch.arange(6890)
    v = data.pos
    e = torch.sparse.FloatTensor(data.edge_index,torch.ones(data.edge_index.shape[1]))
    f = data.face.t()
if args['in_features'] == 1:
    x = torch.ones(len(v),1)
elif args['in_features'] == 3:
    x = v.clone()

out = model(x,v,e,f)
v = (v-v.min())/(v.max()-v.min())
lambda_min = lambda_max = torch.ones(len(model.metric_per_vertex),len(v))
for i,m in enumerate(model.metric_per_vertex):
    weights = torch.ones(len(v))
    for j in range(len(v)):
        metric = m[j].t().mm(m[j])
        eigs = m[j].t().mm(m[j]).eig()[0][:,0]
        lambda_max[i,j] = eigs.max()
        lambda_min[i,j] = eigs.min()
        #weights[j] = (lambda_max*lambda_min)
        weights[j] = lambda_max[i,j]
    weights = weights.detach().numpy()
    weights /= weights.max()
    utils.mesh2ply(to_numpy(v),to_numpy(f),weights,fname=join(out_dir,'{}_m{}.ply'.format(args['dataset'],i)))
pred = torch.nn.functional.softmax(out,dim=1).max(1)[1]
print('{}'.format(pred.eq(gt).sum().float()/len(gt)))

pred = pred/pred.max().float()
utils.mesh2ply(to_numpy(v),to_numpy(f),pred,fname=join(out_dir,'{}_m{}.ply'.format(args['dataset'],i+1)))
pdb.set_trace()    

