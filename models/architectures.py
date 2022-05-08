from typing import NoReturn

import torch
from torch import FloatTensor, LongTensor, nn
from torch.nn import functional as F

from .mesh_pool import MeshPool
from .metric_conv import MetricConv


class MetricConvBlock(nn.Module):
    r"""
    Residual block using ``MetricConv`` layer for forward propagation. The computation may be expressed as:

                                        :math:y=ELU(MetricConv(x)

    :param in_channels: Number of hidden units used in residual layer
    :param out_channels: Number of hidden units used in residual layer
    :param info: Which metric tensor to use
    :param metric_in_feats: Number of input features for ``MetricConv``
    :param metric_n_hidden: Number of hidden parameters for ``MetricConv``
    :param embedding_dim: Embedding dimension of metric tensor used in ``MetricConv``
    :param symmetric: Boolean indicating symmetry of metric used in ``MetricConv``
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        info: str = "tangent",
        metric_n_hidden: int = 32,
        embedding_dim: int = 3,
        symmetric: bool = True,
    ):
        super(MetricConvBlock, self).__init__()
        # Initialize MetricConv CNN operator
        self.conv = MetricConv(
            in_channels,
            out_channels,
            info=info,
            metric_n_hidden=metric_n_hidden,
            embedding_dim=embedding_dim,
            symmetric=symmetric,
        )
        self.nonlinear = nn.ELU()

    def forward(
        self,
        features: FloatTensor,
        vertices: FloatTensor,
        edges: LongTensor,
        faces: LongTensor,
        eps: float = 1e-5,
    ) -> FloatTensor:
        r"""
        :param features: Input features per vertex
        :param vertices: Positions of vectors in \mathbf{R}^3
        :param edges: Edge connectivity of vertices
        :param faces: Face indices of vertices
        :param eps: Small epsilon used to avoid division by 0

        :return: Returns tensor with ``n_hidden`` features for each vertex
        """
        x = self.conv(features, vertices, edges, faces)
        x = (x - x.mean(dim=0)) / (x.std(dim=0) + eps)
        x = self.nonlinear(x)
        self.metric_per_vertex = self.conv.metric_per_vertex
        return x


class MetricResBlock(nn.Module):
    r"""
    Residual block using ``MetricConv`` layer for forward propagation. The computation may be expressed as:

                                        :math:y=0.5*(ELU(MetricConv(x))+x)

    :param n_hidden: Number of hidden units used in residual layer
    :param info: Which metric tensor to use
    :param metric_in_feats: Number of input features for ``MetricConv``
    :param metric_n_hidden: Number of hidden parameters for ``MetricConv``
    :param embedding_dim: Embedding dimension of metric tensor used in ``MetricConv``
    :param symmetric: Boolean indicating symmetry of metric used in ``MetricConv``
    """

    def __init__(
        self,
        n_hidden: int,
        info: str = "tangent",
        metric_n_hidden: int = 32,
        embedding_dim: int = 3,
        symmetric: bool = True,
    ):
        super(MetricResBlock, self).__init__()
        # Initialize MetricConv CNN operator
        self.conv = MetricConv(
            n_hidden,
            n_hidden,
            info=info,
            metric_n_hidden=metric_n_hidden,
            embedding_dim=embedding_dim,
            symmetric=symmetric,
        )
        self.nonlinear = nn.ELU()

    def forward(
        self,
        features: FloatTensor,
        vertices: FloatTensor,
        edges: LongTensor,
        faces: LongTensor,
        eps: float = 1e-5,
    ) -> FloatTensor:
        r"""
        :param features: Input features per vertex
        :param vertices: Positions of vectors in \mathbf{R}^3
        :param edges: Edge connectivity of vertices
        :param faces: Face indices of vertices
        :param eps: Small epsilon used to avoid division by 0

        :return: Returns tensor with ``n_hidden`` features for each vertex
        """
        residual = features.clone()  # Store original features to be added back as the residual
        x = self.conv(features, vertices, edges, faces)
        x = (x - x.mean(dim=0)) / (x.std(dim=0) + eps)
        x = self.nonlinear(x)
        out = (x + residual) / 2  # Add back residual and divide by 2 for average
        self.metric_per_vertex = self.conv.metric_per_vertex
        return out


class MetricResNet(nn.Module):
    r"""
    Simple network with kwargs['n_layers'] ``MetricResBlock`` residual blocks preceded and followed by ``MetricConv`` layers.

    :param in_channels: Number of input features per vertex
    :param out_channels: Number of output features per vertex
    :param \**kwargs: See below
    :Keyword Arguments:
        * n_layers (``int``) -- Number of residual blocks
        * info (``str``) -- type of ``MetricConv`` to use
        * n_hidden (``int``) -- Number of hidden layers used for for residual blocks
        * embedding_dim (``int``) -- Embedding dimension of metric tensor used in ``MetricConv``
    """

    def __init__(self, in_feats: int, out_feats: int, **kwargs):
        super(MetricResNet, self).__init__()
        n_layers = kwargs["n_layers"] if "n_layers" in kwargs.keys() else 8
        info = kwargs["info"] if "info" in kwargs.keys() else "vanilla"
        n_hidden = kwargs["n_hidden"] if "n_hidden" in kwargs.keys() else 64
        embedding_dim = kwargs["embedding_dim"] if "embedding_dim" in kwargs.keys() else 8
        symmetric = kwargs["symmetric"] if "symmetric" in kwargs.keys() else True

        self.conv1 = MetricConv(in_feats, n_hidden, info=info, embedding_dim=embedding_dim,symmetric=symmetric)

        # Instantiate the residual blocks
        res_blocks = []
        for _ in range(n_layers):
            res_blocks.append(MetricResBlock(n_hidden, info, embedding_dim=embedding_dim,symmetric=symmetric))
        self.res_blocks = nn.ModuleList(res_blocks)

        self.conv2 = MetricConv(n_hidden, out_feats, info=info, embedding_dim=embedding_dim,symmetric=symmetric)

        self.nonlinear = nn.ELU()

    def forward(
        self,
        features: FloatTensor,
        vertices: FloatTensor,
        edges: LongTensor,
        faces: LongTensor,
        eps: float = 1e-5,
    ) -> FloatTensor:
        r"""
        :param features: Input features per vertex
        :param vertices: Positions of vectors in \mathbf{R}^3
        :param edges: Edge connectivity of vertices
        :param faces: Face indices of vertices
        :param eps: Small epsilon used to avoid division by 0

        :return: Returns tensor with ``out_feats`` features for each vertex
        """

        self.metric_per_vertex = []

        x = features

        x = self.conv1(x, vertices, edges, faces)
        x = (x - x.mean(dim=0)) / (x.std(dim=0) + eps)
        x = self.nonlinear(x)
        self.metric_per_vertex.append(self.conv1.metric_per_vertex)

        for i in range(len(self.res_blocks)):
            x = self.res_blocks[i](x, vertices, edges, faces)
            self.metric_per_vertex.append(self.res_blocks[i].metric_per_vertex)

        out = self.conv2(x, vertices, edges, faces)
        self.metric_per_vertex.append(self.conv2.metric_per_vertex)

        return out


class LinearMetricNet(nn.Module):
    r"""
    This model adapts the architecture of the correspondence model used in SpiralNet++ (https://leichen2018.github.io/files/spiralnet_plusplus.pdf). We replace the SpiralNet++ convolutional operators with ``MetricConv`` operators.

    :param in_channels: Number of input features per vertex
    :param out_channels: Number of output features per vertex
    :param \**kwargs: See below
    :Keyword Arguments:
        * info (``str``) -- type of ``MetricConv`` to use
        * embedding_dim (``int``) -- Embedding dimension of metric tensor used in ``MetricConv``
    """

    def __init__(self, in_feats: int, out_feats: int, **kwargs):
        super(LinearMetricNet, self).__init__()
        info = kwargs["info"] if "info" in kwargs.keys() else "vanilla"
        embedding_dim = kwargs["embedding_dim"] if "embedding_dim" in kwargs.keys() else 8
        symmetric = kwargs["symmetric"] if "symmetric" in kwargs.keys() else True

        self.fc0 = nn.Linear(3, 16)
        self.conv1 = MetricConv(16, 32, info=info, embedding_dim=embedding_dim,symmetric=symmetric)
        self.conv2 = MetricConv(32, 64, info=info, embedding_dim=embedding_dim,symmetric=symmetric)
        self.conv3 = MetricConv(64, 128, info=info, embedding_dim=embedding_dim,symmetric=symmetric)
        self.fc1 = nn.Linear(128, 256)
        self.fc2 = nn.Linear(256, out_feats)

        self.reset_parameters()

    def reset_parameters(self) -> NoReturn:
        """
        Sets initial parameters of weights follow a Uniform Xavier distribution for fully connected matrices and a uniform distribution for biases.
        """
        nn.init.xavier_uniform_(self.fc0.weight, gain=1)
        nn.init.xavier_uniform_(self.fc1.weight, gain=1)
        nn.init.xavier_uniform_(self.fc2.weight, gain=1)
        nn.init.constant_(self.fc0.bias, 0)
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.constant_(self.fc2.bias, 0)

    def forward(self, features: FloatTensor, vertices: FloatTensor, edges: LongTensor, faces: LongTensor) -> FloatTensor:
        """
        :param features: Input features per vertex
        :param vertices: Positions of vectors in \mathbf{R}^3
        :param edges: Edge connectivity of vertices
        :param faces: Face indices of vertices

        :return: Returns tensor contains ``out_feats`` features for each vertex.
        """
        x = features
        x = F.elu(self.fc0(x))
        x = F.elu(self.conv1(x, vertices, edges, faces))
        x = F.elu(self.conv2(x, vertices, edges, faces))
        x = F.elu(self.conv3(x, vertices, edges, faces))
        x = F.elu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        out = self.fc2(x)
        return out


class MetricConvNet(nn.Module):
    r"""
    All (``MetricConv``) convolutional network using a progressively wider network as described below:

                                in_feats->16->32->64->128->256->out_feats

    :param in_channels: Number of input features per vertex
    :param out_channels: Number of output features per vertex
    :param \**kwargs: See below
    :Keyword Arguments:
        * classification (``bool``) -- Bool indicating whether model is end-to-end or classification
        * info (``str``) -- type of ``MetricConv`` to use
        * embedding_dim (``int``) -- Embedding dimension of metric tensor used in ``MetricConv``
        * layers (``List[int]``) -- Sequence of conv layers to use``
    """

    def __init__(self, in_feats: int, out_feats: int, **kwargs):
        super(MetricConvNet, self).__init__()
        classification = kwargs["classification"] if "classification" in kwargs.keys() else False
        info = kwargs["info"] if "info" in kwargs.keys() else "vanilla"
        embedding_dim = kwargs["embedding_dim"] if "embedding_dim" in kwargs.keys() else 8
        symmetric = kwargs["symmetric"] if "symmetric" in kwargs.keys() else True
        layers = kwargs["layers"] if "layers" in kwargs.keys() else [16, 32, 64, 128, 256]
        layers = [in_feats] + layers

        convs = []
        for i in range(len(layers) - 2):
            if layers[i] == layers[i + 1]:
                conv = MetricResBlock(layers[i], info, embedding_dim=embedding_dim,symmetric=symmetric)
            else:
                conv = MetricConv(layers[i], layers[i + 1], info=info, embedding_dim=embedding_dim,symmetric=symmetric)
            convs.append(conv)

        if classification:
            assert kwargs["num_vertices"] is not None, "To use the classification layer you must specify num_vertices."
            num_vertices = kwargs["num_vertices"]
            # If classification network then instantiate two linear layers used at end of network to convert to logits
            self.fc1 = nn.Linear(layers[-2] * num_vertices, layers[-1])
            self.fc2 = nn.Linear(layers[-1], out_feats)

            self.reset_parameters()
        else:
            if layers[-2] == layers[-1]:
                conv = MetricResBlock(layers[i], info, embedding_dim=embedding_dim,symmetric=symmetric)
            else:
                conv = MetricConv(layers[-2], layers[-1], info=info, embedding_dim=embedding_dim,symmetric=symmetric)
            convs.append(conv)
            convs.append(MetricConv(layers[-1], out_feats, info=info, embedding_dim=embedding_dim,symmetric=symmetric))
        self.convs = nn.ModuleList(convs)

        self.classification = classification
        self.nonlinear = nn.ELU()

    def reset_parameters(self) -> NoReturn:
        """
        Sets initial parameters of weights follow a Uniform Xavier distribution for fully connected matrices and
        a uniform distribution for biases.
        """
        nn.init.xavier_uniform_(self.fc1.weight, gain=1)
        nn.init.xavier_uniform_(self.fc2.weight, gain=1)
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.constant_(self.fc2.bias, 0)

    def forward(
        self,
        features: FloatTensor,
        vertices: FloatTensor,
        edges: LongTensor,
        faces: LongTensor,
        eps: float = 1e-5,
    ) -> FloatTensor:
        r"""
        :param features: Input features per vertex
        :param vertices: Positions of vectors in \mathbf{R}^3
        :param edges: Edge connectivity of vertices
        :param faces: Face indices of vertices
        :param eps: Small epsilon used to avoid division by 0

        :return: Returns tensor with ``out_feats`` features for each vertex
        """
        self.metric_per_vertex = []

        x = features

        for conv in self.convs:
            x = conv(x, vertices, edges, faces)
            x = self.nonlinear(x)
            x = (x - x.mean(dim=0)) / (x.std(dim=0) + eps)
            self.metric_per_vertex.append(conv.metric_per_vertex)

        if self.classification:
            x = x.flatten()
            x = self.fc1(x)
            x = self.nonlinear(x)
            x = F.dropout(x, training=self.training)

            out = self.fc2(x)
        else:
            out = self.conv6(x, vertices, edges, faces)
            self.metric_per_vertex.append(self.conv6.metric_per_vertex)

        return out
