import pytest
import torch

from models import metric_conv

# Create test data (triangular-based pyramid)
v0 = torch.tensor([0, 0, torch.rand(1)])
v1 = torch.tensor([0, torch.rand(1), 0])
v2 = torch.tensor([torch.rand(1), 0, 0])
v3 = torch.tensor([0, -torch.rand(1), 0])

pos = torch.cat((v0.view(1, 3), v1.view(1, 3), v2.view(1, 3), v3.view(1, 3)), dim=0)
edges = torch.tensor(
    [
        [0, 1],
        [0, 2],
        [0, 3],
        [1, 2],
        [1, 3],
        [2, 3],
        [1, 0],
        [2, 0],
        [3, 0],
        [2, 1],
        [3, 1],
        [3, 2],
    ]
).t()
faces = torch.tensor([[0, 1, 2], [0, 2, 3], [0, 1, 3], [1, 3, 2]])

# Model parameters
in_feats = int(torch.randint(1, 100, (1,)))
out_feats = int(torch.randint(1, 100, (1,)))

# Features per vertex
features_per_vertex = torch.rand(4, in_feats)

# Iterate MetricConv using all the different metric types
@pytest.mark.parametrize("info", ["feature", "face", "tangent", "vanilla"])
def test_metric_conv(info: str):
    """
    Test of ``MetricConv`` from the ``metric_conv`` module with the given ``info`` parameter.

    :param info: Which metric to use when computing weighted adjacency matrix for forward pass
    """
    cnn = metric_conv.MetricConv(in_feats, out_feats, info=info)
    out = cnn(features_per_vertex, pos, edges, faces)
    assert out.shape[1] == out_feats
