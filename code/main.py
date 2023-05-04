from torch import nn, optim
from torch.utils.data import DataLoader
import argparse
import wandb
import os

from networks import Linear
from datasets import FeatureDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    # experiment
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--root", type=str, default="/home/frans/Datasets")
    parser.add_argument("--experiment_name", type=str, default=None)
    # wandb
    parser.add_argument("--wandb_project", type=str, default="mscfransdeboer")
    parser.add_argument("--wandb_name", type=str, default=None)
    # data
    parser.add_argument("--dataset", type=str, default="cholec80")
    parser.add_argument("--data_type", type=str, default="features")
    parser.add_argument("--data_dir", type=str, default="features/i3d_embeddings")
    parser.add_argument("--flat", action="store_true")
    parser.add_argument("--bounding_boxes", action="store_true")
    parser.add_argument(
        "--rsd_type", type=str, default="none", choices=["none", "minutes", "seconds"]
    )
    parser.add_argument("--train_split", type=str, default="t12_0.txt")
    parser.add_argument("--test_split", type=str, default="e_0.txt")
    # training
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--iterations", type=int, default=503 * 60)
    # optimizer
    parser.add_argument(
        "--optimizer", type=str, default="adam", choices=["adam", "sgd"]
    )
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    # scheduler
    parser.add_argument("--lr_decay", type=float, default=0.1)
    parser.add_argument("--lr_decay_every", type=int, default=503 * 30)
    # network
    parser.add_argument("--network", type=str, default="ute", choices=["ute"])
    parser.add_argument("--feature_dim", type=int, default=64)
    parser.add_argument("--embed_dim", type=int, default=20)
    # logging
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--test_every", type=int, default=200)

    return parser.parse_args()


def train(network, batch, device, optimizer=None):
    l2_loss = nn.MSELoss(reduction="sum")
    name, data, progress = batch
    data = data.to(device)
    B, _ = data.shape
    predicted_progress = network(data).reshape(B)
    progress = progress.to(device)
    if optimizer:
        optimizer.zero_grad()
        l2_loss(predicted_progress, progress).backward()
        optimizer.step()

    return {
        "l2_loss": l2_loss(predicted_progress, progress),
        "count": B,
    }


def main():
    args = parse_args()
    data_root = os.path.join(args.root, args.dataset)
    experiment_path = None
    if args.experiment_name:
        experiment_path = os.path.join(args.root, "experiments", args.experiment_name)

    wandb.init(
        project="ute",
        name=args.wandb_name,
        config={
            "dataset": args.dataset,
            "data_type": args.data_type,
            "train_split": args.train_split,
            "test_split": args.test_split,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "lr_decay": args.lr_decay,
            "lr_decay_every": args.lr_decay_every,
            "feature_dim": args.feature_dim,
            "embed_dim": args.embed_dim,
            "iterations": args.iterations,
        },
    )

    # TODO: Create Datasets
    # - Feature Dataset (flat / sequential)
    # - Image Dataset (flat / sequential)
    if args.data_type == "features":
        train_set = FeatureDataset(
            data_root, args.data_type, args.train_split, flat=args.flat
        )
        test_set = FeatureDataset(
            data_root, args.data_type, args.test_split, flat=args.flat
        )
    elif args.data_type == "images":
        if "ucf" in args.dataset:
            pass
        else:
            pass
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, num_workers=4, shuffle=True
    )
    test_loader = DataLoader(
        test_set, batch_size=args.batch_size, num_workers=4, shuffle=False
    )
    # TODO: Create network
    # - Features: Linear (flat), RSD (sequential), ProgressNet (sequential)
    # - Images RSD (flat), ProgressNet (flat)
    network = Linear(args.feature_dim, args.embed_dim)
    # TODO: Create optimizer
    optimizer = optim.Adam(
        network.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    # TODO: Seems fine?
    scheduler = optim.lr_scheduler.StepLR(optimizer, args.lr_decay_every, args.lr_decay)

    print("--- Network ---")
    print(network)
    print("--- Datasets ---")
    print(f"Train {len(train_set)} ({len(train_loader)})")
    print(f"Test {len(test_set)} ({len(test_loader)})")


if __name__ == "__main__":
    main()
