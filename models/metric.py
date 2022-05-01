from typing import NoReturn, Tuple

import torch
import torch_scatter as ts
from torch import FloatTensor, LongTensor, nn, sparse


def build_sparse_adjacency(
    edges: LongTensor,
    values: FloatTensor,
    device: str,
    kernel: str = "rational",
    symmetric: bool = True,
    num_vertices: int = None,
    remove_reference: bool = False,
) -> sparse.FloatTensor:
    """
    Creates a sparse, right-stochastic matrix (rows sum to 1) based on given indices and values. The input values are passed through a kernel, after which the output is normalized based on local connectivity.

    :param edges: Edges/connectivity of nodes (size: n_edges x 2)
    :param values: Values associated to each edge (size: n_edges x 1)
    :param device: Device to perform computation on (i.e. cpu or gpu)
    :param kernel: Kernel used to map values to the range (0,1]
    :param symmetric: Boolean to indicate whether or not to symmetrize matrix by adding transpose
    :param remove_reference: Boolean to indicate whether or not to include reference vertices in computed matrix

    :return: Returns sparse adjacency matrix that is used for a convolutional pass
    """
    if num_vertices is None:
        num_vertices = edges.max() + 1

    adj_shape = [num_vertices, num_vertices]

    if kernel == "rational":
        weights = 1 / (1 + values ** 2)  # Map every distance into range (0,1]
    else:
        assert kernel == "gaussian", f'"{kernel}" is not an appropriate kernel. Please choose from ["rational","gaussian"].'

        weights = torch.exp(-(values ** 2))

    if remove_reference:
        # Removes reference node from computation (as well as forward propagation)
        tmp = edges[0] != edges[1]
        edges = edges[:, tmp]
        weights = weights[tmp]

    weights_sum = (
        torch.zeros(num_vertices).to(device).scatter_add(0, edges[0], weights)
    )  # For each ref. vertex add weights of neighbouring nodes
    weights_normalized = weights / weights_sum[edges[0]]  # Normalize weights with respect to 1-ring about each vertex
    adjacency = sparse.FloatTensor(edges, weights_normalized, torch.Size(adj_shape))
    if symmetric:
        # Build another sparse matrix whose rows sum to 1
        weights_sum = (
            torch.zeros(num_vertices).to(device).scatter_add(0, edges[1], weights)
        ) 
        weights_normalized = weights / weights_sum[edges[1]]  
        adjacency_col = sparse.FloatTensor(edges, weights_normalized, torch.Size(adj_shape))

        # Take average of adjacency and transpose of column-normalized adjacency
        adjacency = 0.5 * (adjacency + adjacency_col.transpose(1,0))
    return adjacency


def compute_face_area_and_angle(vertices: FloatTensor, faces: LongTensor, eps: float = 1e-8) -> Tuple[FloatTensor]:
    """
    :param vertices: Vertices
    :param faces: Faces

    :return: Returns total area of all faces incident to each vertex and total angle of all interior angles incident to each vertex.
    """
    device = faces.device

    # We concatenate all permutations of face indices to get all faces incident to each vertex
    f0 = faces.type(LongTensor).to(device)

    f1 = torch.ones(faces.shape[0], 3)
    f1[:, 0] = faces[:, 1]
    f1[:, 1] = faces[:, 0]
    f1[:, 2] = faces[:, 2]
    f1 = f1.type(LongTensor).to(device)

    f2 = torch.ones(faces.shape[0], 3)
    f2[:, 0] = faces[:, 2]
    f2[:, 1] = faces[:, 0]
    f2[:, 2] = faces[:, 1]
    f2 = f2.type(LongTensor).to(device)

    faces_perm = torch.cat((f0, f1, f2), dim=0)

    # Compute tangent vectors from reference vertex along each incident face
    tangent1 = vertices[faces_perm[:, 1]] - vertices[faces_perm[:, 0]]
    tangent2 = vertices[faces_perm[:, 2]] - vertices[faces_perm[:, 0]]

    # Compute area of face as half the norm of the cross product of the two vectors spanning the face
    area_per_face = tangent1.cross(tangent2).norm(dim=1) / 2
    total_area = ts.scatter_add(area_per_face, faces_perm[:, 0], dim=0, out=torch.zeros(vertices.shape[0]).to(device))
    mean_area = ts.scatter_mean(area_per_face, faces_perm[:, 0], dim=0, out=torch.zeros(vertices.shape[0]).to(device))
    std_area = ts.scatter_mean(
        (area_per_face - mean_area[faces_perm[:, 0]]) ** 2,
        faces_perm[:, 0],
        dim=0,
        out=torch.zeros(vertices.shape[0]).to(device),
    )

    # Compute angle by first normalizing each tangent vector and then taking their dot product
    tangent1_norm = tangent1 / (tangent1.norm(p=2, dim=1).unsqueeze(1) + eps)
    tangent2_norm = tangent2 / (tangent2.norm(p=2, dim=1).unsqueeze(1) + eps)
    cos = (tangent1_norm * tangent2_norm).sum(dim=1)

    angle_per_face = torch.acos(cos)
    total_angle = ts.scatter_add(
        angle_per_face,
        faces_perm[:, 0],
        dim=0,
        out=torch.zeros(vertices.shape[0]).to(device),
    )

    return total_area, std_area, total_angle


class MetricMLP(nn.Module):
    """
    Multi-layer Perceptron (MLP) used to build a metric tensor at each vertex.

    :param in_channels: Number of input features
    :param n_hidden: Number of hidden units
    :param embedding_dim: Dimension in which the MLP embeds the tangent vectors
    """

    def __init__(self, in_channels: int, n_hidden: int, embedding_dim: int):
        super(MetricMLP, self).__init__()
        self.metric_mlp = nn.Sequential(
            nn.Linear(in_channels, n_hidden),
            nn.ELU(),
            nn.Linear(n_hidden, embedding_dim * 3),
        )
        self.embedding_dim = embedding_dim
        self.reset_parameters()

    def reset_parameters(self) -> NoReturn:
        for m in self.metric_mlp:
            if type(m) == nn.Linear:
                nn.init.xavier_uniform_(m.weight)

    def forward(self, x: FloatTensor) -> FloatTensor:
        """
        :param x: N x f_in tensor where f_in is the number of features per vertex (N vertices)
        :return: N x embedding_dim x 3 tensor
        """
        out = self.metric_mlp(x)
        return out.reshape(len(x), self.embedding_dim, 3)


class Metric(nn.Module):
    """
    A base class whose forward pass computes an attention matrix for the input vertices and connectivity information. The attention values are computed as local distances which are determined by the metric tensor generated by the ``MetricMLP``.

    For each vertex, the ``MetricMLP`` receives as input the local mesh features computed by the ``compute_features`` method and outputs a corresponding (square-root of) metric tensor.

    Note that the ``compute_features`` method is meant to be over-ridden with the creation of another class which inherits ``Metric`` class. The ``compute_features`` therein can be hand-crafted by the user.

    :param feature_channels: Expected number of input features for each vertex
    :param n_hidden: Number of hidden units in `MetricMLP`
    :param embedding_dim: Dimension that `MetricMLP` embeds the tangent vectors into
    :param symmetric: Boolean deciding whether to symmetrize the attention matrix by adding to it its transpose
    """

    def __init__(
        self,
        feature_channels: int = None,
        n_hidden: int = 32,
        embedding_dim: int = 3,
        symmetric: bool = True,
    ):
        super(Metric, self).__init__()
        self.embedding_dim = embedding_dim
        self.symmetric = symmetric
        self.metric_mlp = None

    def compute_features(
        self,
        features: FloatTensor,
        vertices: FloatTensor,
        edges: LongTensor,
        faces: LongTensor,
    ) -> FloatTensor:
        """
        Function to be over-ridden when using ``Metric`` as base class.

        :param features: Input features per vertex
        :param vertices: Positions of vectors in \mathbf{R}^3
        :param edges: Edge connectivity of vertices
        :param faces: Face indices of vertices

        :return: Local features about each vertex
        """
        return None

    def forward(
        self,
        features: FloatTensor,
        vertices: FloatTensor,
        edges: LongTensor,
        faces: LongTensor,
    ) -> sparse.FloatTensor:
        """
        :param features: Input features per vertex
        :param vertices: Positions of vectors in \mathbf{R}^3
        :param edges: Edge connectivity of vertices
        :param faces: Face indices of vertices

        :return: sparse.FloatTensor containing weighted edges based on non-Euclidean distances.
        """
        device = vertices.device.type
        edges = edges.unique(dim=1)
        tangent_vecs = vertices[edges[1]] - vertices[edges[0]]

        # Compute input features for metric mlp
        metric_input_features = self.compute_features(features, vertices, edges, faces)

        # Compute metric for each vertex using previously computed areas and angles
        self.metric_per_vertex = self.metric_mlp(metric_input_features)

        # Use metric to compute inner products / distance
        diff = torch.matmul(self.metric_per_vertex[edges[0]], tangent_vecs.view(len(edges[0]), 3, 1)).squeeze()
        dist = diff.norm(p=2, dim=1) if self.embedding_dim > 1 else torch.abs(diff)

        # Now build weighted adjacency matrix (attention matrix using distances from metric)
        new_adj = build_sparse_adjacency(edges, dist, device, symmetric=self.symmetric, num_vertices=len(vertices))

        return new_adj


class FaceMetric(Metric):
    """
    Child class of ``Metric`` that uses face information to generate metric features.

    :param feature_channels: Expected number of input features for each vertex
    :param n_hidden: Number of hidden units in `MetricMLP`
    :param embedding_dim: Dimension that `MetricMLP` embeds the tangent vectors into
    :param symmetric: Boolean deciding whether to symmetrize the attention matrix by adding to it its transpose
    """

    def __init__(self, feature_channels: int = None, n_hidden: int = 32, embedding_dim: int = 3, symmetric: bool = True):
        super(FaceMetric, self).__init__(feature_channels, n_hidden, embedding_dim, symmetric)

        # Features (per vertex) are total area of each face + total interior angle about each vertex
        in_channels = 3
        self.metric_mlp = MetricMLP(in_channels, n_hidden, embedding_dim)

    def compute_features(
        self,
        features: FloatTensor,
        vertices: FloatTensor,
        edges: LongTensor,
        faces: LongTensor,
    ) -> FloatTensor:
        """
        Computes features to be passed into a ``MetricConv`` module.
        The features computed are the total area of all incident faces about each vertex, as well as the total interior angle.

        :param features: Input features per vertex
        :param vertices: Positions of vectors in \mathbf{R}^3
        :param edges: Edge connectivity of vertices
        :param faces: Face indices of vertices

        :return: Local features about each vertex
        """

        total_area, std_area, total_angle = compute_face_area_and_angle(vertices, faces)
        total_area = total_area / total_area.max()
        std_area = std_area / std_area.max()
        total_angle = total_angle / total_angle.abs().max()
        metric_input_features = torch.cat([total_angle.unsqueeze(1), total_area.unsqueeze(1), std_area.unsqueeze(1)], dim=1)

        return metric_input_features


class FeatureMetric(Metric):
    """
    Child class of ``Metric`` that uses feature information to generate metric features.

    :param feature_channels: Expected number of input features for each vertex
    :param n_hidden: Number of hidden units in `MetricMLP`
    :param embedding_dim: Dimension that `MetricMLP` embeds the tangent vectors into
    :param symmetric: Boolean deciding whether to symmetrize the attention matrix by adding to it its transpose
    """

    def __init__(self, feature_channels: int = None, n_hidden: int = 32, embedding_dim: int = 3, symmetric: bool = True):
        super(FeatureMetric, self).__init__(feature_channels, n_hidden, embedding_dim, symmetric)

        assert feature_channels is not None, "You must specify the number of features per vertex using feature_channels"
        # Features (per vertex) are xyz position + mean of tangent of features + std of tangent of features
        in_channels = 3 + 2 * feature_channels
        self.metric_mlp = MetricMLP(in_channels, n_hidden, embedding_dim)

    def compute_features(
        self, features: FloatTensor, vertices: FloatTensor, edges: LongTensor, faces: LongTensor, eps: float = 1e-8
    ) -> FloatTensor:
        """
        Computes features to be passed into a ``MetricConv`` module.
        The features computed are the mean and standard deviations of the differences of the features between each reference node and its neighbours.

        :param features: Input features per vertex
        :param vertices: Positions of vectors in \mathbf{R}^3
        :param edges: Edge connectivity of vertices
        :param faces: Face indices of vertices
        :param eps: Small epsilon used to avoid potential division by 0

        :return: Local features about each vertex
        """

        feature_vecs = features[edges[1]] - features[edges[0]]
        feature_vecs = feature_vecs / (feature_vecs.norm(p=2, dim=1).unsqueeze(1) + eps)
        mu = ts.scatter_mean(feature_vecs, edges[0], dim=0, out=torch.full_like(features, 0))
        sigma = (
            ts.scatter_mean(
                (feature_vecs - mu[edges[0]]) ** 2,
                edges[0],
                dim=0,
                out=torch.full_like(features, 0),
            )
            + eps
        )

        metric_input_features = torch.cat([vertices, mu, sigma], dim=1)

        return metric_input_features


class TangentMetric(Metric):
    """
    Child class of ``Metric`` that uses tangent information to generate metric features.

    :param feature_channels: Expected number of input features for each vertex
    :param n_hidden: Number of hidden units in `MetricMLP`
    :param embedding_dim: Dimension that `MetricMLP` embeds the tangent vectors into
    :param symmetric: Boolean deciding whether to symmetrize the attention matrix by adding to it its transpose
    """

    def __init__(self, feature_channels: int = None, n_hidden: int = 32, embedding_dim: int = 3, symmetric: bool = True):
        super(TangentMetric, self).__init__(feature_channels, n_hidden, embedding_dim, symmetric)

        # Features (per vertex) are mean of tangent vectors + std of tangent vectors
        in_channels = 6
        self.metric_mlp = MetricMLP(in_channels, n_hidden, embedding_dim)

    def compute_features(
        self, features: FloatTensor, vertices: FloatTensor, edges: LongTensor, faces: LongTensor, eps: float = 1e-8
    ) -> FloatTensor:
        """
        Computes features to be passed into a ``MetricConv`` module.
        The features computed are the mean and standard deviation of all tangent vectors about each vertex.

        :param features: Input features per vertex
        :param vertices: Positions of vectors in \mathbf{R}^3
        :param edges: Edge connectivity of vertices
        :param faces: Face indices of vertices
        :param eps: Small epsilon used to avoid potential division by 0

        :return: Local features about each vertex
        """

        tangent_vecs = vertices[edges[1]] - vertices[edges[0]]
        mu = ts.scatter_mean(tangent_vecs, edges[0], dim=0, out=torch.full_like(vertices, 0))
        sigma = (
            ts.scatter_mean(
                (tangent_vecs - mu[edges[0]]) ** 2,
                edges[0],
                dim=0,
                out=torch.full_like(vertices, 0),
            )
            + eps
        )
        metric_input_features = torch.cat([mu, sigma], dim=1)

        return metric_input_features


class VanillaMetric(nn.Module):
    """
    Constructs a module whose forward pass computes an attention matrix for the input vertices and connectivity information. The weights of the matrix are based on the inverse of the Euclidean distances to neighbouring nodes (i.e. nodes further away have less weight/attention). The attention weights are normalize about each vertex over all neighbouring nodes.

    :param symmetric: Boolean deciding whether to symmetrize the attention matrix by adding to it its transpose
    """

    def __init__(self, symmetric: bool = True):
        super(VanillaMetric, self).__init__()
        self.symmetric = symmetric

    def forward(
        self, features: FloatTensor, vertices: FloatTensor, edges: LongTensor, faces: LongTensor
    ) -> sparse.FloatTensor:
        """
        :param features: Input features per vertex
        :param vertices: Positions of vectors in \mathbf{R}^3
        :param edges: Edge connectivity of vertices
        :param faces: Face indices of vertices

        :return: sparse.FloatTensor containing weighted edges based on Euclidean distances.
        """
        device = vertices.device.type
        num_nodes = len(vertices)
        self.metric_per_vertex = torch.eye(3, 3).unsqueeze(0).repeat(num_nodes, 1, 1)
        edges = edges.unique(dim=1)

        tangent_vecs = vertices[edges[1]] - vertices[edges[0]]

        dist = tangent_vecs.norm(p=2, dim=1)

        # Now build weighted adjacency matrix (attention matrix using distances from metric)
        new_adj = build_sparse_adjacency(edges, dist, device, symmetric=self.symmetric, num_vertices=num_nodes)
        return new_adj
