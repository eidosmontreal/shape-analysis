import argparse
import os
import pdb
from os.path import join
from typing import NoReturn

import numpy
import torch
import torch_geometric
import torch_geometric.transforms as T

import datasets
import utils

device = "cuda" if torch.cuda.is_available() else "cpu"


def to_numpy(x: torch.Tensor) -> numpy.array:
    """
    Convert tensor on GPU to a numpy array on CPU

    :param x: PyTorch tensor

    :return: Numpy array
    """
    return x.detach().cpu().numpy()


def visualize_metric(log_dir: str, data_dir: str, target_dir: str, data_idx: int = 1) -> NoReturn:
    """
    Generates visualizations of MetricConv modules in a trained architecture over a mesh.
    The mesh is drawn from the same dataset that the model was trained on.

    :param log_dir: Directory containing trained model and experiment information
    :param data_dir: Directory containing dataset
    :param target_dir: Directory to store visualizations
    :param data_idx: Index of dataset to visualize upon
    """

    model, args = utils.get_model(log_dir, True)  # Load model found in log_dir and load trained parameters

    if "HumanSegmentation" in args["dataset"]:
        ds = getattr(datasets, "HumanSegmentation")(join("data", args["dataset"]), train=False)
        (v, e, f), gt = ds[data_idx]
    elif "COSEG" in args["dataset"]:
        ds = getattr(datasets, "COSEG")(join("data", args["dataset"]), classes=args["classes"], train=False)
        (v, e, f), gt = ds[data_idx]
    elif args["dataset"] == "FAUST":
        ds = torch_geometric.datasets.FAUST(join("data", "FAUST"), False, transform=T.FaceToEdge(False))
        data = ds[data_idx]
        v = data.pos
        e = data.edge_index
        f = data.face.t()
        gt = torch.arange(len(v))
    if args["in_channels"] == 1:
        x = torch.ones(len(v), 1)
    elif args["in_channels"] == 3:
        x = v.clone()

    out = model(x, v, e, f)
    v = (v - v.min()) / (v.max() - v.min())
    lambda_min = lambda_max = torch.ones(len(model.metric_per_vertex), len(v))
    for i, m in enumerate(model.metric_per_vertex):
        weights = torch.ones(len(v))
        for j in range(len(v)):
            metric = m[j].t().mm(m[j])
            eigs = metric.eig()[0][:, 0]
            lambda_max[i, j] = eigs.max()
            lambda_min[i, j] = eigs.min()
            weights[j] = lambda_max[i, j]
        weights = weights.detach().numpy()
        weights /= weights.max()
        utils.mesh2ply(
            to_numpy(v),
            to_numpy(f),
            weights,
            fname=join(target_dir, f"{args['dataset']}_{data_idx}_m{i}.ply"),
        )
    pred = torch.nn.functional.softmax(out, dim=1).max(1)[1]
    print(pred.eq(gt).sum().float() / len(gt))

    pred = pred / pred.max().float()
    utils.mesh2ply(
        to_numpy(v),
        to_numpy(f),
        pred,
        fname=join(target_dir, f"{args['dataset']}_{data_idx}_pred.ply"),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log-dir",
        type=str,
        required=True,
        help="Root directory contains pretrained weights and experiments args",
    )
    parser.add_argument("--data-dir", type=str, required=True, help="Directory containing data")
    parser.add_argument(
        "--target-dir",
        type=str,
        default=None,
        required=False,
        help="Directory for storing sample meshes",
    )
    parser.add_argument(
        "--data-idx",
        type=int,
        default=1,
        required=False,
        help="Index of data to visualize",
    )
    args = parser.parse_args()

    if args.target_dir is None:
        args.target_dir = join(args.log_dir, "samples")
        os.makedirs(args.target_dir, exist_ok=True)
    visualize_metric(args.log_dir, args.data_dir, args.target_dir, args.data_idx)
