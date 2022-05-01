from typing import NoReturn

import torch
from torch import FloatTensor, LongTensor, nn, sparse

from .metric import FaceMetric, FeatureMetric, TangentMetric, VanillaMetric

info_to_metric = {"face": FaceMetric, "feature": FeatureMetric, "tangent": TangentMetric}


class MetricConv(nn.Module):
    """
    ``MetricConv`` is a convolutional operator for mesh and graph structured data that incorporated elements from pseudo-Riemannian geometry. In particular, the message-passing/connectivity/adjacency matrix is computed using non-Euclidean distances, which are computed dynamically with a metric that changes between vertices (i.e. the kernel is not static across the mesh).

     In particular, ``MetricConv`` performs a localized convolution operation that uses a dynamic weighted adjacency matrix, which can be expressed as:
                                                     :math:Y=AXW+b

     where,
         A - adjacency (connectivity) matrix used for message passing/aggregating information
         X - input features (dense [n_verts x in_channels] matrix)
         W - filter (dense [in_channels x out_feats] matrix)
         b - bias

     A is computed using the metric indicated by the `info` parameter.

     :param in_channels: Number of input features given to `MetricMLP`.
     :param out_feats: Number of output features per vertex`
     :param bias: Boolean to decide whether or not CNN computation will add a bias vector
     :param info: Which metric to use when computing weighted adjacency matrix for forward pass
     :param metric_n_hidden: Number of hidden units used in `MetricMLP`
     :param embedding_dim: Embedding dimension of tangent vectors for `MetricMLP`
     :param symmetric: Boolean to decide whether weighted adjacency matrix will be symmetric or not
    """

    def __init__(
        self,
        in_channels: int,
        out_feats: int,
        bias: bool = True,
        info: str = "face",
        metric_n_hidden: int = 32,
        embedding_dim: int = 3,
        symmetric: bool = True,
    ):
        super(MetricConv, self).__init__()

        self.info = info
        self.in_channels = in_channels
        self.out_feats = out_feats
        self.weights = nn.Parameter(torch.Tensor(in_channels, out_feats))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_feats))
        else:
            self.register_parameter("bias", None)

        if info in info_to_metric.keys():
            self.metric = info_to_metric[info](in_channels, metric_n_hidden, embedding_dim, symmetric)
            # self.metric = Metric(info, in_channels, metric_n_hidden, embedding_dim, symmetric)
        elif info == "vanilla":
            # Metric tensor is identity across all vertices
            self.metric = VanillaMetric(symmetric)
        else:
            raise ValueError(f"{info} is not an available metric type")

        self.reset_parameters()

    def reset_parameters(self) -> NoReturn:
        "Glorot initialization"
        nn.init.xavier_uniform_(self.weights)
        if self.bias is not None:
            self.bias.data.uniform_(-0.1, 0.1)

    def forward(self, features: FloatTensor, vertices: FloatTensor, edges: LongTensor, faces: LongTensor) -> FloatTensor:
        """
        :param features: Input features per vertex
        :param vertices: Positions of vectors in \mathbf{R}^3
        :param edges: Edge connectivity of vertices
        :param faces: Face indices of vertices

        :return: Tensor containing features computed from computation described above.
        """
        # Construct weighted adjacency matrix based on connectivity, using prescribed metric.
        weighted_adj = self.metric(features, vertices, edges, faces)
        self.weighted_adj = weighted_adj

        # Store the metric tensor at each vertex for each layer
        self.metric_per_vertex = self.metric.metric_per_vertex

        # Compute output features via standard message-passing operation
        out = sparse.mm(weighted_adj, torch.mm(features, self.weights))
        if self.bias is not None:
            out += self.bias
        return out
