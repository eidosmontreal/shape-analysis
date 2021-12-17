import numpy as np

import torch
from torch import nn
from torch import LongTensor, FloatTensor, sparse
from torch.nn import functional as F

import torch_scatter as ts

from typing import Tuple
"""
TODO: - add batch-sizes (for now training uses accumlated gradients)
"""

def build_sparse_adjacency(idx: LongTensor, values: FloatTensor, device: str, kernel: str='rational', symmetric: bool = True, remove_reference: bool = False) -> sparse.FloatTensor:
    """ 
    Creates a sparse, right-stochastic matrix (rows sum to 1) based on given indices and values. The input values are passed through a kernel, after which the output is normalized based on local connectivity.

    :param idx: Edges/connectivity of nodes (size: n_edges x 2)
    :param values: Values associated to each edge (size: n_edges x 1)
    :param device: Device to perform computation on (i.e. cpu or gpu) 
    :param kernel: Kernel used to map values to the range (0,1]
    :param symmetric: Boolean to indicate whether or not to symmetrize matrix by adding transpose
    :param remove_reference: Boolean to indicate whether or not to include reference vertices in computed matrix
    """
    n_verts = idx.max()+1
    adj_shape = [n_verts,n_verts]
    if kernel == 'rational':
        weights = 1/(1+values**2) # Map every distance into range (0,1]
    else: 
        assert kernel == 'gaussian', '"{}" is not an appropriate kernel. Please choose from ["rational","gaussian"].'.format(kernel)
        weights = torch.exp(-values**2)
    
    if remove_reference:
        # Removes reference node from computation (as well as forward propagation)
        tmp = (idx[0] != idx[1])
        idx = idx[:,tmp]
        weights = weights[tmp]
    
    weights_sum = torch.zeros(n_verts).to(device).scatter_add(0,idx[0],weights) # For each ref. vertex add weights of neighbouring nodes
    weights = weights/weights_sum[idx[0]] # Normalize weights with respect to 1-ring about each vertex
    adjacency = sparse.FloatTensor(idx,weights,torch.Size(adj_shape))
    if symmetric:
        # Take average of adjacency and its transpose
        adjacency = 0.5*(adjacency + adjacency.transpose(1,0))
    return adjacency

def compute_face_area_and_angle(pos: FloatTensor, faces: LongTensor, eps: float=1e-8) -> Tuple[FloatTensor]:
    """
    Returns total area of all faces incident to each vertex and total angle of all interior angles incident to each vertex.

    :param pos: Vertices
    :param faces: Faces
    """
    device = faces.device
    
    # We concatenate all permutations of face indices to get all faces incident to each vertex
    f0 = faces.type(LongTensor).to(device)
    
    f1 = torch.ones(faces.shape[0],3)
    f1[:,0] = faces[:,1]
    f1[:,1] = faces[:,0]
    f1[:,2] = faces[:,2]
    f1 = f1.type(LongTensor).to(device)

    f2 = torch.ones(faces.shape[0],3)
    f2[:,0] = faces[:,2]
    f2[:,1] = faces[:,0]
    f2[:,2] = faces[:,1]
    f2 = f2.type(LongTensor).to(device)

    faces_perm = torch.cat((f0,f1,f2),dim=0)
    
    # Compute tangent vectors from reference vertex along each incident face
    tangent1 = pos[faces_perm[:,1]] - pos[faces_perm[:,0]]
    tangent2 = pos[faces_perm[:,2]] - pos[faces_perm[:,0]]

    # Compute area of face as half the norm of the cross product of the two vectors spanning the face
    area_per_face = tangent1.cross(tangent2).norm(dim=1)/2
    total_area = ts.scatter_add(area_per_face,faces_perm[:,0],dim=0,out=torch.zeros(pos.shape[0]).to(device))
    
    # Compute angle by first normalizing each tangent vector and then taking their dot product
    tangent1_norm = tangent1/(tangent1.norm(p=2,dim=1).unsqueeze(1)+eps)
    tangent2_norm = tangent2/(tangent2.norm(p=2,dim=1).unsqueeze(1)+eps)
    cos = (tangent1_norm*tangent2_norm).sum(dim=1)
    
    angle_per_face = torch.acos(cos)
    total_angle = ts.scatter_add(angle_per_face,faces_perm[:,0],dim=0,out=torch.zeros(pos.shape[0]).to(device))

    return total_area, total_angle

class MetricMLP(nn.Module):
    """
    MLP used to build a metric tensor at each vertex.

    :param in_feats: Number of input features
    :param n_hidden: Number of hidden units
    :param embedding_dim: Dimension in which the MLP embeds the tangent vectors

    """
    def __init__(self,in_feats: int, n_hidden: int, embedding_dim: int):
        super(MetricMLP,self).__init__()
        self.metric_mlp = nn.Sequential(nn.Linear(in_feats,n_hidden),
                                                nn.ELU(),
                                                nn.Linear(n_hidden,embedding_dim*3))
        self.embedding_dim = embedding_dim
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.metric_mlp:
            if type(m) == nn.Linear:
                nn.init.xavier_uniform_(m.weight)

    def forward(self,x: FloatTensor) -> FloatTensor:
        """
        :param x: N x f_in tensor where f_in is the number of features per vertex (N vertices)
        :return: N x embedding_dim x 3 tensor
        """
        out = self.metric_mlp(x)
        return out.reshape(len(x),self.embedding_dim,3)

class FaceMetric(nn.Module):
    """
    Constructs a module whose forward pass computes an attention matrix for the input vertices and connectivity information. The attention values are computed as  local distances which are determined by the metric tensor generated by the ``MetricMLP``. 
    
    For each vertex, the ``MetricMLP`` receives as input the total area of the all incident faces and total angle of all interior angles and outputs a corresponding (square-root of) the metric tensor.

    :param in_feats: Number of input features given to `MetricMLP`. 
    :param n_hidden: Number of hidden units in `MetricMLP`
    :param embedding_dim: Dimension that `MetricMLP` embeds the tangent vectors into
    :param symmetric: Boolean deciding whether to symmetrize the attention matrix by adding to it its transpose
    """
    def __init__(self,in_feats: int, n_hidden: int=32, embedding_dim: int=3, symmetric: bool=True):
        super(FaceMetric,self).__init__()
        self.embedding_dim = embedding_dim
        self.symmetric = symmetric
        self.metric_mlp = MetricMLP(in_feats,n_hidden,embedding_dim)

    def forward(self,features: FloatTensor, pos: FloatTensor, edges: LongTensor, faces: LongTensor ,eps: float = 1e-8) -> sparse.FloatTensor:
        """
        :param features: Input features per vertex
        :param pos: Positions of vectors in \mathbf{R}^3
        :param edges: Edge connectivity of vertices
        :param faces: Face indices of vertices

        :return: sparse.FloatTensor containing weighted edges based on non-Euclidean distances.
        """
        device = pos.device.type
        idx = edges.unique(dim=1)
        tangent_vecs = pos[idx[1]] - pos[idx[0]]
       
        # Compute angles and areas for each vertex
        total_area, total_angle = compute_face_area_and_angle(pos,faces)
        total_angle = total_angle/total_angle.abs().max()
        total_area = total_area/total_area.max()
        metric_input_features = torch.cat([total_angle.unsqueeze(1),total_area.unsqueeze(1)],dim=1)

        # Compute metric for each vertex using previously computed areas and angles 
        metric_per_vertex = self.metric_mlp(metric_input_features)

        # Use metric to compute inner products / distance
        diff = torch.matmul(metric_per_vertex[idx[0]],tangent_vecs.view(len(idx[0]),3,1)).squeeze()
        dist = diff.norm(p=2,dim=1) if self.embedding_dim > 1 else torch.abs(diff)
       
        # Now build weighted adjacency matrix (attention matrix using distances from metric)
        
        new_adj = build_sparse_adjacency(idx,dist,device,symmetric=self.symmetric)
        self.metric_per_vertex = metric_per_vertex
        return new_adj

class FeatureMetric(nn.Module):
    """
    Constructs a module whose forward pass computes an attention matrix for the input vertices and connectivity information. The attention values are computed as  local distances which are determined by the metric tensor generated by the ``MetricMLP``. 
    
    For each vertex, the ``MetricMLP`` receives as input the total area of the position as well as the mean and variance of the features vectors for all neighbouring nodes, and outputs a corresponding (square-root of) the metric tensor.

    :param in_feats: Number of input features given to `MetricMLP`. 
    :param n_hidden: Number of hidden units in `MetricMLP`
    :param embedding_dim: Dimension that `MetricMLP` embeds the tangent vectors into
    :param symmetric: Boolean deciding whether to symmetrize the attention matrix by adding to it its transpose
    """
    def __init__(self,in_feats: int, n_hidden: int=32, embedding_dim: int=3, symmetric: bool=True):
        super(FeatureMetric,self).__init__()
        self.embedding_dim = embedding_dim
        self.symmetric = symmetric
        self.metric_mlp = MetricMLP(in_feats,n_hidden,embedding_dim)

    def forward(self,features: FloatTensor, pos: FloatTensor, edges: LongTensor, faces: LongTensor ,eps: float = 1e-8) -> sparse.FloatTensor:
        """
        :param features: Input features per vertex
        :param pos: Positions of vectors in \mathbf{R}^3
        :param edges: Edge connectivity of vertices
        :param faces: Face indices of vertices

        :return: sparse.FloatTensor containing weighted edges based on Euclidean distances.
        """
        device = pos.device.type
        idx = edges.unique(dim=1)
        tangent_vecs = pos[idx[1]] - pos[idx[0]]
        
        # Compute mean and std of vectors from reference point
        feature_vecs = features[idx[1]] - features[idx[0]]
        feature_vecs = feature_vecs/(feature_vecs.norm(p=2,dim=1).unsqueeze(1)+eps)
        
        mu = ts.scatter_mean(feature_vecs,idx[0],dim=0,out=torch.full_like(features,0))
        sigma = ts.scatter_mean((feature_vecs-mu[idx[0]])**2,idx[0],dim=0,out=torch.full_like(features,0)) + eps
        
        metric_input_features = torch.cat([pos,mu,sigma],dim=1)
        
        # Compute respective metrics per vertex based on position, and mean,std of available direction vectors
        metric_per_vertex = self.metric_mlp(metric_input_features)

        # Use metric to compute inner products / distance
        diff = torch.matmul(metric_per_vertex[idx[0]],tangent_vecs.view(len(idx[0]),3,1)).squeeze()
        dist = diff.norm(p=2,dim=1) if self.embedding_dim > 1 else torch.abs(diff)
       
        # Now build weighted adjacency matrix (attention matrix using distances from metric)
        new_adj = build_sparse_adjacency(idx,dist,device,symmetric=self.symmetric)
        
        self.metric_per_vertex = metric_per_vertex
        return new_adj


class TangentMetric(nn.Module):
    """
    Constructs a module whose forward pass computes an attention matrix for the input vertices and connectivity information. The attention values are computed as  local distances which are determined by the metric tensor generated by the ``MetricMLP``. 
    
    For each vertex, the ``MetricMLP`` receives as input the mean and variance of all incident tangent vectors and outputs a corresponding (square-root of) the metric tensor.

    :param in_feats: Number of input features given to `MetricMLP`. 
    :param n_hidden: Number of hidden units in `MetricMLP`
    :param embedding_dim: Dimension that `MetricMLP` embeds the tangent vectors into
    :param symmetric: Boolean deciding whether to symmetrize the attention matrix by adding to it its transpose
    """
    def __init__(self,in_feats: int, n_hidden: int=32, embedding_dim: int=3, symmetric: bool=True):
        super(TangentMetric,self).__init__()
        self.embedding_dim = embedding_dim
        self.symmetric = symmetric
        self.metric_mlp = MetricMLP(in_feats,n_hidden,embedding_dim)

    def forward(self,features: FloatTensor, pos: FloatTensor, edges: LongTensor, faces: LongTensor ,eps: float = 1e-8) -> sparse.FloatTensor:
        """
        :param features: Input features per vertex
        :param pos: Positions of vectors in \mathbf{R}^3
        :param edges: Edge connectivity of vertices
        :param faces: Face indices of vertices

        :return: sparse.FloatTensor containing weighted edges based on non-Euclidean distances.
        """
        device = pos.device.type
        idx = edges.unique(dim=1)
        tangent_vecs = pos[idx[1]] - pos[idx[0]]
        
        # Compute mean and std of vectors from reference point
        mu = ts.scatter_mean(tangent_vecs,idx[0],dim=0,out=torch.full_like(pos,0))
        sigma = ts.scatter_mean((tangent_vecs-mu[idx[0]])**2,idx[0],dim=0,out=torch.full_like(pos,0)) + eps

        metric_input_features = torch.cat([mu,sigma],dim=1)
        
        # Compute respective metrics per vertex based on position, and mean,std of available direction vectors
        metric_per_vertex = self.metric_mlp(metric_input_features)

        # Use metric to compute inner products / distance
        diff = torch.matmul(metric_per_vertex[idx[0]],tangent_vecs.view(len(idx[0]),3,1)).squeeze()
        dist = diff.norm(p=2,dim=1) if self.embedding_dim > 1 else torch.abs(diff)
       
        # Now build weighted adjacency matrix (attention matrix using distances from metric)
        new_adj = build_sparse_adjacency(idx,dist,device,symmetric=self.symmetric)

        self.metric_per_vertex = metric_per_vertex
        return new_adj

class VanillaMetric(nn.Module):
    """
    Constructs a module whose forward pass computes an attention matrix for the input vertices and connectivity information. The weights of the matrix are based on the inverse of the Euclidean distances to neighbouring nodes (i.e. nodes further away have less weight/attention). The attention weights are normalize about each vertex over all neighbouring nodes.

    :param symmetric: Boolean deciding whether to symmetrize the attention matrix by adding to it its transpose
    """
    def __init__(self,symmetric: bool=True):
        super(VanillaMetric,self).__init__()
        self.symmetric = symmetric

    def forward(self,features: FloatTensor, pos: FloatTensor, edges: LongTensor, faces: LongTensor ,eps: float = 1e-8) -> sparse.FloatTensor:
        """
        :param features: Input features per vertex
        :param pos: Positions of vectors in \mathbf{R}^3
        :param edges: Edge connectivity of vertices
        :param faces: Face indices of vertices

        :return: sparse.FloatTensor containing weighted edges based on Euclidean distances.
        """
        device = pos.device.type
        num_nodes = len(pos)
        self.metric_per_vertex = torch.eye(3,3).unsqueeze(0).repeat(num_nodes,1,1)
        idx = edges.unique(dim=1)
        
        tangent_vecs = pos[idx[1]] - pos[idx[0]]

        dist = tangent_vecs.norm(p=2,dim=1)
       
        # Now build weighted adjacency matrix (attention matrix using distances from metric)
        new_adj = build_sparse_adjacency(idx,dist,device,symmetric=self.symmetric)
        return new_adj

class MetricConv(nn.Module):
    """
   ``MetricConv`` is a convolutional operator for mesh and graph structured data that incorporated elements from pseudo-Riemannian geometry. In particular, the message-passing/connectivity/adjacency matrix is computed using non-Euclidean distances, which are computed dynamically with a metric that changes between vertices (i.e. the kernel is not static across the mesh).

    In particular, ``MetricConv`` performs a localized convolution operation that uses a dynamic weighted adjacency matrix, which can be expressed as: 
                                                    :math:Y=AXW+b 
    
    where,
        A - adjacency (connectivity) matrix used for message passing/aggregating information
        X - input features (dense [n_verts x in_feats] matrix)
        W - filter (dense [in_feats x out_feats] matrix)
        b - bias

    A is computed using the metric indicated by the `info` parameter.
    
    :param in_feats: Number of input features given to `MetricMLP`. 
    :param out_feats: Number of output features per vertex`
    :param bias: Boolean to decide whether or not CNN computation will add a bias vector
    :param info: Which metric to use when computing weighted adjacency matrix for forward pass
    :param metric_n_hidden: Number of hidden units used in `MetricMLP`
    :param embedding_dim: Embedding dimension of tangent vectors for `MetricMLP`
    :param symmetric: Boolean to decide whether weighted adjacency matrix will be symmetric or not
    """
    def __init__(self,in_feats: int, out_feats: int ,bias: bool=True,
                    info: str='face', metric_n_hidden: int=32, embedding_dim: int=3, symmetric: bool=True):
        super(MetricConv,self).__init__()

        self.info = info
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.weights = nn.Parameter(torch.Tensor(in_feats,out_feats))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_feats))
        else:
            self.register_parameter('bias', None)
        
        if info == 'feature':
            # Uses position and mean/variance of difference in features among neighbouring vertices to construct metric tensors
            metric_in_feats = 3 + 2*in_feats
            self.metric = FeatureMetric(metric_in_feats,metric_n_hidden,embedding_dim,symmetric) 
        elif info == 'face':
            # Uses total area and angle of vertex to construct metric tensor
            metric_in_feats = 2
            self.metric = FaceMetric(metric_in_feats,metric_n_hidden,embedding_dim,symmetric) 
        elif info == 'tangent':
            # Uses mean and variance of vertces' tangent vectors (given my neighbouring vertices) to construct metric tensor
            metric_in_feats = 6 
            self.metric = TangentMetric(metric_in_feats,metric_n_hidden,embedding_dim,symmetric) 
        else:
            assert info == 'vanilla', '{} is not an available metric type'.format(info) 
            # Metric tensor is identity across all vertices
            self.metric = VanillaMetric(symmetric)

        self.reset_parameters()
    
    def reset_parameters(self):
       'Glorot initialization'
       nn.init.xavier_uniform_(self.weights)
       if self.bias is not None:
           self.bias.data.uniform_(-.1, .1)

    def forward(self,x: FloatTensor, pos: FloatTensor, 
                    edges: LongTensor, faces: LongTensor) -> FloatTensor:
        """
        :param features: Input features per vertex
        :param pos: Positions of vectors in \mathbf{R}^3
        :param edges: Edge connectivity of vertices
        :param faces: Face indices of vertices

        :return: Tensor containing features computed from computation described above.
        """
        # Construct weighted adjacency matrix based on connectivity, using prescribed metric.
        weighted_adj = self.metric(x,pos,edges,faces)
        
        # Store the metric tensor at each vertex for each layer
        self.metric_per_vertex = self.metric.metric_per_vertex

        # Compute output features via standard message-passing operation
        out = sparse.mm(weighted_adj,torch.mm(x,self.weights))
        if self.bias is not None:
            out += self.bias
        return out
