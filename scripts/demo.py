import argparse
import os
from os.path import join
from typing import NoReturn

import torch
import yaml
from torch_geometric import transforms
from torch_geometric.data import DataLoader
from torch_geometric.datasets import FAUST

import datasets
import utils


def test_model(root: str, data_dir: str, target_dir: str, n_samples: int = 3) -> NoReturn:
    """
    Tests trained model from a given experiment.

    :param root: Directory where experiment parameters and trained weights are stored
    :param data_dir: Directory where data is stored
    :param target_dir: Directory for storing samples
    :param n_samples: Number of examples to test on
    """
    os.makedirs(target_dir, exist_ok=True)
    config = yaml.safe_load(open("config.yml"))

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, train_args = utils.get_model(args.root, True)
    model.eval()
    model = model.to(device)

    utils.print_dict(train_args)

    data_root = join(data_dir, train_args["dataset"])
    if train_args["dataset"] in config["datasets"]["segmentation"]:
        if train_args["dataset"] == "COSEG":
            dataset = getattr(datasets, train_args["dataset"])(data_root, classes=train_args["classes"], train=False)
            dataloader = torch.utils.data.DataLoader(dataset, shuffle=False)
            target_max = 4.0
        else:
            dataset = getattr(datasets, train_args["dataset"])(data_root, train=False)
            dataloader = torch.utils.data.DataLoader(dataset, shuffle=False)
            target_max = 7.0
    elif train_args["dataset"] == "FAUST":
        dataset = FAUST(data_root, train=False, transform=transforms.FaceToEdge(False))
        dataloader = DataLoader(dataset, shuffle=False, batch_size=1)
        target_max = dataset[0].num_nodes

    for i, data in enumerate(dataloader):
        if train_args["dataset"] in config["datasets"]["segmentation"]:
            (v, e, f), target = data
            v = v[0].to(device)
            e = e[0].to(device)
            f = f[0].to(device)
            target = target[0].to(device)

            if train_args["in_channels"] == 1:
                x = torch.ones((len(v), 1)).to(device)
            elif train_args["in_channels"] == 3:
                x = v.clone()
        elif train_args["dataset"] == "FAUST":
            v = data.pos.to(device)
            e = data.edge_index.to(device)
            f = data.face.t().to(device)
            target = torch.arange(target_max).to(device)

            x = data.pos.to(device)

        out = model(x, v, e, f)
        pred = torch.nn.functional.softmax(out, dim=1).max(1)[1]
        accuracy = pred.eq(target).sum().float() / len(target)
        print("%.3f accuracy for sample %d." % (float(accuracy), i))

        utils.mesh2ply(
            v.cpu().numpy(),
            f.cpu().numpy(),
            weights=pred.cpu().numpy() / target_max,
            fname=join(args.target_dir, f"pred{i}.ply"),
        )
        utils.mesh2ply(
            v.cpu().numpy(),
            f.cpu().numpy(),
            weights=target.cpu().numpy() / target_max,
            fname=join(args.target_dir, f"gt{i}.ply"),
        )
        if i + 1 == args.n_samples:
            break


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=str,
        required=True,
        help="Root directory contains pretrained weights and experiments args",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Directory containing data",
    )
    parser.add_argument(
        "--target-dir",
        type=str,
        default=None,
        required=False,
        help="Directory for storing sample meshes.",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=3,
        required=False,
        help="Number of samples to demo on",
    )
    args = parser.parse_args()

    if args.target_dir is None:
        args.target_dir = join(args.root, "samples")

    test_model(args.root, args.data_dir, args.target_dir, args.n_samples)
