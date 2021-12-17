import torch
from torch import FloatTensor

from typing import List

def orthonormality_penalization(matrices: List[FloatTensor]) -> FloatTensor:
    """
    Returns the squared L2 distance between the square of each matrix and the identity, across each each item in the input list and each vertex in each tensor. 
    In other words, if M is a metric tensor at a vertex, the following is calculated:
                                        |M^TM - I|^2
    
    :param: List of N x n x m tensors, where N is the number of vertices and n x m is the size of the metric tensor at each vertex.
    """
    device = matrices[0].device
    id_dim = torch.matmul(matrices[0].transpose(1,2),matrices[0]).shape[-1]
    
    penalty = 0
    for m in matrices:
        metric_tensor = torch.matmul(m.transpose(1,2),m) 
        penalty += torch.norm(torch.matmul(metric_tensor,metric_tensor) - torch.eye(id_dim).view(1,id_dim,id_dim).to(device))**2
    penalty = penalty/(len(matrices) * matrices[0].shape[0])  
    return penalty

def frobenius_norm_penalization(matrices: List[FloatTensor]) -> FloatTensor:
    """
    Returns the squared L2 distance between each matrix and the identity, across each each item in the input list and each vertex in each tensor. 
    In other words, if M is a metric tensor at a vertex, the following is calculated:
                                        |M - I|^2

    :param: List of N x n x m tensors, where N is the number of vertices and n x m is the size of the metric tensor at each vertex.
    """
    device = matrices[0].device
    id_dim = torch.matmul(matrices[0].transpose(1,2),matrices[0]).shape[-1]
    
    penalty = 0
    for m in matrices:
        metric_tensor = torch.matmul(m.transpose(1,2),m) 
        penalty += metric_tensor.norm(p=2)**2 
    penalty = penalty/(len(matrices) * matrices[0].shape[0])  
    return penalty
