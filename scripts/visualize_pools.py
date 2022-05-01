import argparse
import os
from os.path import join
from typing import NoReturn

from torch_geometric import transforms
from torch_geometric.datasets import FAUST

import datasets
import utils


def visualize_pools(
    log_dir: str, data_dir: str, target_dir: str = None, num_samples: int = 1, train: bool = False
) -> NoReturn:
    if target_dir is None:
        target_dir = join(log_dir, "samples")
    os.makedirs(target_dir, exist_ok=True)

    model, args = utils.get_model(log_dir, load_params=True)

    if args["dataset"] == "COSEG":
        ds = datasets.COSEG(join(data_dir, "COSEG"), classes=args["classes"], train=train)
    elif args["dataset"] == "FAUST":
        transform = transforms.Compose([transforms.FaceToEdge(False), transforms.AddSelfLoops()])
        ds = FAUST(join(data_dir, "FAUST"), train=train, transform=transform)

    for i in range(num_samples):
        verts = []
        if args["dataset"] == "FAUST":
            data = ds[i]
            v = data.pos
            f = data.face.t()
            e = data.edge_index
        else:
            (v, e, f) = ds[i][0]
        model(v, v, e, f)

        j = 0
        for (x, v, e, f, cluster) in model.mesh_per_layer:
            # _,v,e,f,__ = pool(torch.rand(len(v)),v,e,f.type(torch.long).t())
            # f = f.t()
            utils.mesh2obj(v.numpy(), f.numpy(), fname=join(target_dir, f"mesh{i}_pool{j}.obj"))
            verts.append(len(v))
            j += 1
        print(f"Mesh {i+1}: {verts}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log-dir", type=str, required=True, help="Root directory contains pretrained weights and experiments args"
    )

    parser.add_argument("--data-dir", type=str, required=True, help="Directory containing data")
    parser.add_argument("--target-dir", type=str, default=None, required=False, help="Directory for storing sample meshes")

    parser.add_argument("--num-samples", type=int, default=1, required=False, help="Number of samples to render")

    args = parser.parse_args()
    visualize_pools(args.log_dir, args.data_dir, args.target_dir, args.num_samples)
