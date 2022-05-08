import os
import time
from os.path import join
from typing import Tuple, Type

import numpy as np
import torch
import train_args
import yaml
from torch import FloatTensor, nn, optim
from torch.nn import functional
from torch.utils.data import DataLoader
from tqdm import tqdm

import datasets
import models
import utils


def train(
    model: Type[nn.Module],
    train_loader: Type[DataLoader],
    optimizer: Type[optim.Optimizer],
    iteration=0,
    scheduler: Type[FloatTensor] = None,
    batch_size: int = 1,
    in_channels: int = 3,
    orthonormality_penalty: float = 0.0,
    frobenius_penalty: float = 0.0,
) -> Tuple[nn.Module, optim.Optimizer, FloatTensor, int]:
    """
    Trains model on data loaded by ``train_loader`` using ``optimizer``.

    :param model: Model to be trained
    :param train_loader: Loader for the training data
    :param optimizer: Optimizer used to optimize model's weights
    :param iteration: Iteration count of number of backprops
    :param scheduler: Schedule of learning rates for optimizer
    :param batch_size: Number of items in batch
    :param in_channels: Number of input features for model
    :param orthonormality_penalty: Weight of orthonormality penalization
    :param frobenius_penalty: Weight of frobenius penalization

    :return: Returns updated model, optimizer, averaged loss over all iterations, and iteration count
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Run training iterations to optimize weights
    loss_epoch = torch.zeros(len(train_loader))
    idx = 0

    # TRAINING LOOP
    model.train()
    for (v, e, f), target in train_loader:
        v = v[0].to(device)
        e = e[0].to(device)
        f = f[0].to(device)
        target = target[0].to(device)

        if in_channels == 1:
            input_features = torch.ones((len(v), 1)).to(device)
        elif in_channels == 3:
            # Use positions as features
            input_features = v
        pred = model(input_features, v, e, f)

        # Compute losses
        loss = functional.nll_loss(functional.log_softmax(pred, dim=1), target)
        if orthonormality_penalty > 0:
            loss += utils.orthonormality_penalization(model.metric_per_vertex)
        if frobenius_penalty > 0:
            loss += utils.frobenius_norm_penalization(model.metric_per_vertex)

        # Backpropagate loss
        loss = loss / batch_size
        loss.backward()

        if (iteration + 1) % batch_size == 0:
            # Every [batch-size] iteration(s) update weights
            optimizer.step()
            optimizer.zero_grad()
            if scheduler is not None:
                for param in optimizer.param_groups:
                    param["lr"] = scheduler[int((iteration + 1) / batch_size - 1)]

        loss_epoch[idx] = loss.cpu().data
        iteration += 1
        idx += 1

    loss_mean = loss_epoch.mean()
    loss_std = loss_epoch.std()

    return model, optimizer, loss_mean, loss_std, iteration


def test(model: Type[nn.Module], test_loader: Type[DataLoader], in_channels: int = 3) -> FloatTensor:
    """
    Tests given ``model`` with ``test_loader``.

    :param model: Model to be tested
    :param test_loader: ``DataLoader`` loading test data
    :param in_channels: Number of input features per vertex for model

    :return: Test accuracy
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    acc_epoch = torch.zeros(len(test_loader))
    loss_epoch = torch.zeros(len(test_loader))
    idx = 0

    # TESTING LOOP
    model.eval()
    for (v, e, f), target in test_loader:
        v = v[0].to(device)
        e = e[0].to(device)
        f = f[0].to(device)
        target = target[0].to(device)

        if in_channels == 1:
            in_features = torch.ones((len(v), 1)).to(device)
        elif in_channels == 3:
            in_features = v

        pred = model(in_features, v, e, f)

        loss_epoch[idx] = functional.nll_loss(functional.log_softmax(pred, dim=1), target).cpu().data
        acc_epoch[idx] = (functional.softmax(pred, dim=1).max(1)[1].eq(target).sum().float() / len(target)).cpu().data

        idx += 1

    loss_mean = loss_epoch.mean()
    loss_std = loss_epoch.std()

    acc_mean = acc_epoch.mean()
    acc_std = acc_epoch.std()
    return loss_mean, loss_std, acc_mean, acc_std


if __name__ == "__main__":
    root = join(os.path.dirname(os.path.realpath(__file__)), "..")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Parse input arguments
    parser = train_args.get_parser()
    args = parser.parse_args()
    if args.yaml is not None:
        args = yaml.safe_load(open(args.yaml))
    else:
        assert args.model is not None and args.dataset is not None, "Indicate which model and dataset to use."
        args = dict(vars(args))
    args["time_stamp"] = time.strftime("%m_%d_%H%M%S")

    if args["dir_name"] is None:
        args["dir_name"] = args["time_stamp"]

    # Set seed
    torch.manual_seed(args["seed"])

    # Create logging files/folders for losses
    log_dir = join(root, "logs", args["dataset"], "segmentation", args["dir_name"])
    os.makedirs(log_dir, exist_ok=True)
    epoch_log = open(join(log_dir, "train_test.csv"), "w")
    print("epoch,train_loss_mean,train_loss_std,test_loss_mean,test_loss_std,acc_mean,acc_std", file=epoch_log, flush=True)

    # Load dataset and define dataloader
    if "COSEG" in args["dataset"]:
        train_dataset = datasets.COSEG(join(root, "data", args["dataset"]), args["classes"], train=True)
        test_dataset = datasets.COSEG(join(root, "data", args["dataset"]), args["classes"], train=False)
    elif "HumanSegmentation" in args["dataset"]:
        train_dataset = datasets.HumanSegmentation(join(root, "data", args["dataset"]), train=True)
        test_dataset = datasets.HumanSegmentation(join(root, "data", args["dataset"]), train=False)
    else:
        raise Exception(f"{args['datasets']} is not an available dataset.")

    train_loader = DataLoader(train_dataset, shuffle=True)
    test_loader = DataLoader(test_dataset, shuffle=False)
    print(f"Datasets loaded. ({len(train_dataset)} in train set, {len(test_dataset)} in test set)")

    # Load model
    model = getattr(models, args["model"])(args["in_channels"], args["out_channels"], **args)
    model = model.to(device)
    args["num_parameters"] = utils.num_parameters(model)
    print(f"Model loaded. ({args['num_parameters']} parameters)")

    # Define optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=args["lr"], betas=(0.9, 0.99), weight_decay=args["tikhonov"])
    num_gradient_steps = int((len(train_dataset) / args["batch_size"]) * args["n_epochs"])
    if args["scheduler"] == "step":
        scheduler = args["lr"] * args["decay_rate"] ** torch.linspace(0, num_gradient_steps, 1)
    elif args["scheduler"] == "cosine":
        scheduler = (
            args["min_lr"] + (args["lr"] - args["min_lr"]) * (torch.cos(torch.linspace(0, np.pi, num_gradient_steps)) + 1) / 2
        )
    else:
        scheduler = None

    # Dump experiment args in a yaml
    utils.print_dict(args)
    with open(join(log_dir, "args.yaml"), "w") as f:
        yaml.dump(args, f)

    # Training / test loop
    iteration = 0
    training_loop = tqdm(range(args["n_epochs"]))
    for epoch in training_loop:

        model, optimizer, train_loss_mean, train_loss_std, iteration = train(
            model,
            train_loader,
            optimizer,
            iteration,
            scheduler,
            args["batch_size"],
            args["in_channels"],
            args["orthonormality_penalty"],
            args["frobenius_penalty"],
        )
        test_loss_mean, test_loss_std, acc_mean, acc_std = test(model, test_loader, args["in_channels"])

        print(
            "%d,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f"
            % (epoch, train_loss_mean, train_loss_std, test_loss_mean, test_loss_std, acc_mean, acc_std),
            file=epoch_log,
            flush=True,
        )
        training_loop.set_description("loss=%.4f,acc=%.4f" % (float(train_loss_mean), float(acc_mean)))

        # Plot training/test data
        if epoch % args["snapshot_freq"] == 0:
            utils.plot_train_test([log_dir])
            torch.save(
                {
                    "weights": model.state_dict(),
                    "epoch": epoch,
                    "train_loss": train_loss_mean,
                    "accuracy": acc_mean,
                },
                join(log_dir, "experiment.pth"),
            )

    epoch_log.close()
    print("Training complete.")
