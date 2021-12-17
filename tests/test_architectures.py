import torch

from models import architectures

import pytest

# Create test data (triangular-based pyramid)
v0 = torch.tensor([0, 0, torch.rand(1)])
v1 = torch.tensor([0, torch.rand(1), 0])
v2 = torch.tensor([torch.rand(1), 0, 0])
v3 = torch.tensor([0, -torch.rand(1), 0])

pos = torch.cat((v0.view(1, 3), v1.view(1, 3), v2.view(1, 3), v3.view(1, 3)), dim=0)
edges = torch.tensor([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3], [1, 0], [2, 0], [3, 0], [2, 1], [3, 1], [3, 2]]).t()
faces = torch.tensor([[0, 1, 2], [0, 2, 3], [0, 1, 3], [1, 3, 2]])

# Define model parameters
in_feats = int(torch.randint(1, 100, (1,)))
out_feats = int(torch.randint(1, 100, (1,)))

kwargs = {
    "n_hidden": int(torch.randint(10, 100, (1,))),
    "info": "face",
    "embedding_dim": int(torch.randint(1, 10, (1,))),
    "n_layers": int(torch.randint(1, 20, (1,))),
    "num_vertices": 4,
}

# Features per vertex
features_per_vertex = torch.rand(4, in_feats)  # Each vertex has 3 features


def test_metric_resnet():
    model = architectures.MetricResNet(in_feats, out_feats, **kwargs)
    out = model(features_per_vertex, pos, edges, faces)


@pytest.mark.parametrize("classification", [True, False])
def test_metric_conv_net(classification):
    kwargs["classification"] = classification
    model = architectures.MetricConvNet(in_feats, out_feats, **kwargs)
    out = model(features_per_vertex, pos, edges, faces)
    if classification:
        assert out.shape[0] == out_feats
    else:
        assert out.shape[1] == out_feats
    kwargs.pop("classification")


def test_linear_metric_net():
    model = architectures.LinearMetricNet(in_feats, out_feats, **kwargs)
    out = model(pos, edges, faces)
    assert out.shape[1] == out_feats
