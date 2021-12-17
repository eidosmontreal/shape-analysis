import matplotlib

matplotlib.use("Agg")

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import torch_geometric
import torch_geometric.transforms as T
import torch_geometric.datasets as datasets

import pdb, os, time, argparse, json
from os.path import join
from tqdm import tqdm

import utils
import models

import train_args

parser = train_args.get_parser()
args = parser.parse_args()
args = dict(vars(args))

torch.manual_seed(args["seed"])

args["time_stamp"] = time.strftime("%m_%d_%H%M%S")

root = join(os.path.dirname(os.path.realpath(__file__)), "..")
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load dataset
if args["dataset"] in ["ModelNet10", "ModelNet40"]:
    name = args["dataset"].split("ModelNet")[1]
    pre_transform = T.Compose(
        [
            T.SamplePoints(
                args["num_vertices"], remove_faces=False, include_normals=True
            ),
            T.KNNGraph(),
        ]
    )
    train_dataset = getattr(datasets, "ModelNet")(
        join("data", args["dataset"]),
        name=name,
        train=True,
        pre_transform=pre_transform,
    )
    test_dataset = getattr(datasets, "ModelNet")(
        join("data", args["dataset"]),
        name=name,
        train=False,
        pre_transform=pre_transform,
    )

elif args["dataset"] == "ShapeNet":

    class SamplePoints(object):
        """
        Sample a point cloud uniformly.
        """

        def __init__(self, num_vertices):
            self.num_vertices = num_vertices

        def __call__(self, data):
            num = len(data.pos)
            idx = torch.randperm(num)[: self.num_vertices]
            data.pos = data.pos[idx]
            return data

    pre_transform = T.Compose([SamplePoints(args["num_vertices"]), T.KNNGraph()])
    train_dataset = getattr(datasets, args["dataset"])(
        join("data", args["dataset"]), split="train", pre_transform=pre_transform
    )
    test_dataset = getattr(datasets, args["dataset"])(
        join("data", args["dataset"]), split="test", pre_transform=pre_transform
    )

train_loader = torch_geometric.data.DataLoader(train_dataset, shuffle=True)
test_loader = torch_geometric.data.DataLoader(test_dataset, shuffle=False)
print(
    "Datasets loaded. ({} in train set, {} in test set)".format(
        len(train_dataset), len(test_dataset)
    )
)

# Load model
model = getattr(models, args["model"])(
    args["in_features"],
    args["out_features"],
    args["num_vertices"],
    args["embedding_dim"],
)
model = model.to(device)
print("Model loaded. ({} parameters)".format(utils.num_parameters(model)))

# Define optimizer and scheduler
optimizer = torch.optim.Adam(
    model.parameters(), lr=args["lr"], betas=(0.9, 0.99), weight_decay=args["tikhonov"]
)
num_gradient_steps = int((len(train_dataset) / args["batch_size"]) * args["n_epochs"])
if args["scheduler"] == "step":
    scheduler = args["lr"] * args["decay_rate"] ** torch.linspace(
        0, num_gradient_steps, 1
    )
elif args["scheduler"] == "cosine":
    scheduler = (
        args["min_lr"]
        + (args["lr"] - args["min_lr"])
        * (torch.cos(torch.linspace(0, np.pi, num_gradient_steps)) + 1)
        / 2
    )

n_epochs = args["n_epochs"]
bsz = args["batch_size"]

# Create logging files/folders for losses
if args["dir_name"] is None:
    args["dir_name"] = args["time_stamp"]
log_dir = join(root, "logs", args["dataset"], "classification", args["dir_name"])
os.makedirs(log_dir, exist_ok=True)
epoch_log = open(join(log_dir, "train.csv"), "w")
print("epoch,loss", file=epoch_log, flush=True)
training_log = open(join(log_dir, "train_iterations.csv"), "w")
print("epoch,iteration,loss", file=training_log, flush=True)
test_log = open(join(log_dir, "test.csv"), "w")
print("epoch,accuracy", file=test_log, flush=True)

# Dump experiment args in a json
iteration = 0
utils.print_dict(args)
with open(join(log_dir, "args.json"), "w") as f:
    json.dump(args, f)

# Run training iterations to optimize weights
sched_it = 0
training_loop = tqdm(range(n_epochs))
for e in training_loop:
    loss_epoch = 0
    iteration = 0

    model.train()
    num_pts = 0
    for data in train_loader:
        V = data.pos.to(device)
        if len(V) != args["num_vertices"]:
            num_pts += 1
            continue
        E = torch.sparse.FloatTensor(
            data.edge_index,
            torch.ones(data.edge_index.shape[1]),
            torch.Size([args["num_vertices"], args["num_vertices"]]),
        ).to(device)
        target = (
            data.y if args["dataset"] in ["ModelNet10", "ModelNet40"] else data.category
        )

        pred = model(V, V, E)
        loss = -F.log_softmax(pred, dim=1)[0][target]
        (loss / bsz).backward()

        if (iteration + 1) % bsz == 0:
            # Every [batch-size] iterations update weights
            optimizer.step()
            optimizer.zero_grad()
            if args["scheduler"] is not None:
                for param in optimizer.param_groups:
                    param["lr"] = scheduler[sched_it]
                sched_it += 1

        print(
            "%d,%d,%.5f" % (e, iteration, loss.cpu().data.numpy()),
            file=training_log,
            flush=True,
        )
        loss_epoch += loss.cpu().data.numpy()
        iteration += 1

    loss_epoch /= iteration
    print("%d,%.5f" % (e, loss_epoch), file=epoch_log, flush=True)
    print("{} unavailable in training set.".format(num_pts))
    ## Test model
    accuracy = 0
    iteration_test = 0

    model.eval()
    num_pts = 0
    for data in test_loader:
        V = data.pos.to(device)
        if len(V) != args["num_vertices"]:
            num_pts += 1
            continue
        E = torch.sparse.FloatTensor(
            data.edge_index, torch.ones(data.edge_index.shape[1])
        ).to(device)
        target = (
            data.y if args["dataset"] in ["ModelNet10", "ModelNet40"] else data.category
        )
        target = target.to(device)

        pred = model(V, V, E)
        label = F.softmax(pred, dim=1).max(1)[1]
        accuracy += (label == target).float()
        iteration_test += 1

    accuracy /= iteration_test
    print("%d,%.5f" % (e, accuracy), file=test_log, flush=True)
    print("{} unavailable in test set.".format(num_pts))

    if e % 10 == 0:
        utils.plot_all_iterations(log_dir)
        torch.save({"weights": model.state_dict()}, join(log_dir, "weights.pth"))

    training_loop.set_description("%.4f,%.4f" % (float(loss_epoch), float(accuracy)))

training_log.close()
epoch_log.close()
print("Training complete.")
