from typing import Tuple, Union

import torch
import torch_scatter as ts
from torch import FloatTensor, LongTensor

"""
This code has been adapted from the EdgePool code from PyTorch Geometric:
https://github.com/rusty1s/pytorch_geometric/blob/master/torch_geometric/nn/pool/edge_pool.py
"""


class MeshPool(torch.nn.Module):
    r"""The edge pooling operator from the `"Towards Graph Pooling by Edge
    Contraction" <https://graphreason.github.io/papers/17.pdf>`_ and
    `"Edge Contraction Pooling for Graph Neural Networks"
    <https://arxiv.org/abs/1905.10990>`_ papers.

    There are two types of merge operations: `vertices` and `edges`. Each merge operation dictates how mesh downpooling is performed: in the case of vertices, vertex neighbourhoods are collapsed; in the case of edges, edges are collapsed (i.e. pairs of vertices). Clusters are generated based on collapsed vertices.

    :param in_channels: Number of input channels per node
    :param aggr: Aggregation method for downsampling (i.e. how to pool nodes)
    :param factor: Maximum factor to downsample by
    :param merge: Information that will be merged (i.e. edges or vertices)
    """

    def __init__(
        self, factor: int = 2, in_channels: int = 0, aggr: str = "add", merge: str = "vertices", add_self_loop: bool = True
    ):
        super(MeshPool, self).__init__()
        assert merge in ["vertices", "edges"], f"{merge} is not an acceptable merge-type"
        assert aggr in ["mean", "add"], f"{aggr} is not an acceptable aggregation type. Please choose from ['mean','add']."
        self.factor = factor
        self.aggr = aggr
        self.merge = merge
        self.add_self_loop = add_self_loop

        self.mlp = self.get_mlp(in_channels)

    def get_mlp(self, in_channels: int) -> Union[None, torch.nn.Module]:
        """
        Returns None or nn.Module (MLP) for score computation.

        :param in_channels: Number of input channels per item

        :return: None or MLP used to extract vertex/edge scores
        """
        if in_channels == 0:
            return None
        else:
            return torch.nn.Sequential(Linear(2, 1), nn.LeakyReLU())

    def forward(
        self,
        x: FloatTensor,
        vertices: FloatTensor,
        edges: LongTensor,
        faces: LongTensor,
        pool_features: FloatTensor,
        adjacency: torch.sparse.FloatTensor,
    ) -> Tuple[FloatTensor, FloatTensor, LongTensor, LongTensor, FloatTensor]:
        """
        Downpools mesh based on scores. Typing of downpooling is chosen beforehand.

        :param x: Input features per vertex
        :param vertices: Positions of vectors in R^3
        :param edges: Edge connectivity of vertices
        :param faces: Face indices of vertices

        :return: Returns merged (downpooled) features, vertices, edges, faces. Also returns cluster info indicating which vertices were merged together. The cluster info can be used to upsample vertices to original resolution.
        """
        if self.mlp is None:
            assert len(pool_features.squeeze().shape) == 1, "pool_features must be a (len(vertices) x 1) vector"
            scores = pool_features
        else:
            scores = self.mlp(pool_features)

        if self.merge == "vertices":
            raw_new_x, raw_new_vertices, cluster = self.__merge_vertices__(x, vertices, edges, faces, scores, adjacency)
        elif self.merge == "edges":
            cluster = self.__merge_edges__(x, vertices, edges, faces, scores)

        # We compute the new features as an addition of the old ones.
        new_x = getattr(ts, "scatter_" + self.aggr)(raw_new_x, cluster, dim=0)
        new_vertices = getattr(ts, "scatter_" + self.aggr)(raw_new_vertices, cluster, dim=0)

        # Find all edges which DON'T have repeated nodes
        eidx = cluster[edges]
        idx = eidx[0, :] != eidx[1, :]
        new_edges = eidx[:, idx]
        if self.add_self_loop:
            self_loop = torch.arange(len(new_vertices)).unsqueeze(0)
            self_loop = torch.cat([self_loop, self_loop], dim=0).to(new_edges.device)
            new_edges = torch.cat([self_loop, new_edges], dim=1)

        # Find all faces which DON'T have repeated nodes (equiv. to finding all faces that are adjacent to collapsed faces)
        fidx = cluster[faces]
        idx1 = fidx[0, :] != fidx[1, :]
        idx2 = fidx[0, :] != fidx[2, :]
        idx3 = fidx[1, :] != fidx[2, :]
        idx = idx1 & idx2 & idx3
        new_faces = fidx[:, idx].t()

        return new_x, new_vertices, new_edges, new_faces, cluster

    def __merge_vertices__(
        self,
        x: FloatTensor,
        vertices: FloatTensor,
        edges: LongTensor,
        faces: LongTensor,
        vertex_scores: FloatTensor,
        adjacency: torch.sparse.FloatTensor,
    ) -> FloatTensor:
        """
        Downpools mesh based on _vertex_ scores. Typing of downpooling is chosen beforehand.

        :param x: Input features per vertex
        :param vertices: Positions of vectors in R^3
        :param edges: Edge connectivity of vertices
        :param faces: Face indices of vertices
        :param vertex_scores: Scores assigned to each vertex

        :return: FloatTensor that represents cluster of input vertices to new vertices (i.e. relabelling of vertices that indicates which vertices get merged)
        """
        num_vertices = len(vertices)
        target_num_vertices = num_vertices // self.factor
        nodes_remaining = set(range(num_vertices))

        cluster = torch.zeros(num_vertices).type(torch.LongTensor)
        vertices_argsort = torch.argsort(vertex_scores, descending=True)

        # Iterate through all faces, selecting it if it is not incident to
        # another already chosen face.
        i = 0
        raw_new_vertices = vertices.clone()
        raw_new_x = x.clone()
        for idx in vertices_argsort.tolist():
            if idx not in nodes_remaining:
                continue

            nbrs = edges[1, edges[0] == idx]  # Includes self-loops (i.e. idx)
            nbrs = set(nbrs.tolist())
            nbrs_remaining = nbrs.copy()

            index_score = []
            for n in nbrs:
                if n in nodes_remaining:
                    index_score.append([n, adjacency[idx, n]])
                    nodes_remaining.remove(n)

            if len(index_score) == 1:
                # If all neighbours of idx have been merged, record individual cluster and continue
                cluster[idx] = i
                i += 1
                continue

            index_score = torch.tensor(index_score).t()
            nbrs_remaining = index_score[0].long()
            scores = index_score[1] / index_score[1].sum()  # Normalize remaining scores to sum to 1
            for score, n in zip(scores, nbrs_remaining):
                raw_new_x[n] *= score
                raw_new_vertices[n] *= score
                cluster[n] = i

            i += 1

            if len(nodes_remaining) <= target_num_vertices:
                break

        # The remaining nodes are simply kept.
        for node_idx in nodes_remaining:
            cluster[node_idx] = i
            i += 1
        cluster = cluster.to(x.device)
        return raw_new_x, raw_new_vertices, cluster

    def __merge_edges__(
        self, x: FloatTensor, vertices: FloatTensor, edges: LongTensor, faces: LongTensor, edge_scores: FloatTensor
    ) -> FloatTensor:
        """
        Downpools mesh based on _edge_ scores. Typing of downpooling is chosen beforehand.

        :param x: Input features per vertex
        :param vertices: Positions of vectors in R^3
        :param edges: Edge connectivity of vertices
        :param faces: Face indices of vertices
        :param edge_scores: Scores assigned to each edge

        :return: FloatTensor that represents cluster of input vertices to new vertices (i.e. relabelling of vertices that indicates which vertices get merged)
        """
        num_vertices = len(vertices)
        target_num_vertices = num_vertices // self.factor
        nodes_remaining = set(range(num_vertices))

        cluster = torch.zeros(num_vertices).type(torch.LongTensor)
        edges_argsort = torch.argsort(edge_scores, descending=True)

        # Iterate through all faces, selecting it if it is not incident to
        # another already chosen face.
        i = 0
        for idx in edges_argsort.tolist():
            source = int(edges[0, idx])
            if source not in nodes_remaining:
                continue
            target = int(edges[1, idx])
            if target not in nodes_remaining:
                continue
            cluster[source] = i
            cluster[target] = i
            nodes_remaining.remove(source)
            nodes_remaining.remove(target)
            if len(nodes_remaining) <= target_num_vertices:
                break
            i += 1

        # The remaining nodes are simply kept.
        for node_idx in nodes_remaining:
            cluster[node_idx] = i
            i += 1
        cluster = cluster.to(x.device)
        return cluster

    def __repr__(self):
        return "{}({})".format(self.__class__.__name__, self.in_channels)
