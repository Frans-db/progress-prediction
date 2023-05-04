from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
import argparse
import wandb
import torch
import os

from networks import Linear
from networks import ProgressNet, ProgressNetFlat
from networks import RSDNet, RSDNetFlat
from datasets import FeatureDataset, ImageDataset, UCFDataset
from experiment import Experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    # experiment
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--experiment_name", type=str, default=None)
    # wandb
    parser.add_argument("--wandb_project", type=str, default="mscfransdeboer")
    parser.add_argument("--wandb_name", type=str, default=None)
    parser.add_argument("--wandb_tags", type=str, nargs="+")
    parser.add_argument("--wandb_disable", action="store_true")
    # data
    parser.add_argument("--dataset", type=str, default="cholec80")
    parser.add_argument("--data_dir", type=str, default="features/i3d_embeddings")
    parser.add_argument("--flat", action="store_true")
    parser.add_argument("--bboxes", action="store_true")
    parser.add_argument("--subsample", action="store_true")
    parser.add_argument(
        "--rsd_type", type=str, default="none", choices=["none", "minutes", "seconds"]
    )
    parser.add_argument('--fps', type=int, default=None)
    parser.add_argument("--train_split", type=str, default="t12_0.txt")
    parser.add_argument("--test_split", type=str, default="v_0.txt")
    # training
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--iterations", type=int, default=503 * 60)
    # network
    parser.add_argument(
        "--network", type=str, default="progressnet", choices=["ute", "progressnet", "rsdnet"]
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default=None,
        choices=["vgg16", "resnet18", "resnet152"],
    )
    parser.add_argument("--load_backbone", type=str, default=None)
    # network parameters
    parser.add_argument("--feature_dim", type=int, default=64)
    parser.add_argument("--embed_dim", type=int, default=1024)
    parser.add_argument("--dropout_chance", type=float, default=0.5)
    parser.add_argument("--pooling_layers", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--roi_size", type=int, default=4)
    # network loading
    parser.add_argument("--load_experiment", type=str, default=None)
    parser.add_argument("--load_iteration", type=int, default=None)
    # optimizer
    parser.add_argument(
        "--optimizer", type=str, default="adam", choices=["adam", "sgd"]
    )
    parser.add_argument(
        "--loss", type=str, default="l2", choices=["l2", "l1", "smooth_l1"]
    )
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    # scheduler
    parser.add_argument("--lr_decay", type=float, default=0.1)
    parser.add_argument("--lr_decay_every", type=int, default=503 * 30)
    # logging
    parser.add_argument("--log_every", type=int, default=250)
    parser.add_argument("--test_every", type=int, default=1000)

    parser.add_argument("--print_only", action="store_true")

    return parser.parse_args()


def train_flat_features(network, criterion, batch, device, optimizer=None):
    l2_loss = nn.MSELoss(reduction="sum")
    l1_loss = nn.MSELoss(reduction="sum")
    _, data, progress = batch
    data = data.to(device)
    B, _ = data.shape
    predicted_progress = network(data).reshape(B)
    progress = progress.to(device)
    if optimizer:
        optimizer.zero_grad()
        criterion(predicted_progress, progress).backward()
        optimizer.step()

    return {
        "l2_loss": l2_loss(predicted_progress, progress).item(),
        "l1_loss": l1_loss(predicted_progress, progress).item(),
        "count": B,
    }


def train_flat_frames(network, criterion, batch, device, optimizer=None):
    l2_loss = nn.MSELoss(reduction="sum")
    l1_loss = nn.MSELoss(reduction="sum")
    _, frames, progress = batch
    frames = frames.to(device)
    B, _, _, _ = frames.shape
    predicted_progress = network(frames).reshape(B)
    progress = progress.to(device)
    if optimizer:
        optimizer.zero_grad()
        criterion(predicted_progress, progress).backward()
        optimizer.step()

    return {
        "l2_loss": l2_loss(predicted_progress, progress).item(),
        "l1_loss": l1_loss(predicted_progress, progress).item(),
        "count": B,
    }

def train_flat_bbox_frames(network, criterion, batch, device, optimizer=None):
    l2_loss = nn.MSELoss(reduction="sum")
    l1_loss = nn.MSELoss(reduction="sum")
    _, frames, boxes, progress = batch
    frames = frames.to(device)
    boxes = boxes.to(device)
    B, _, _, _ = frames.shape
    predicted_progress = network(frames, boxes).reshape(B)
    progress = progress.to(device)
    if optimizer:
        optimizer.zero_grad()
        criterion(predicted_progress, progress).backward()
        optimizer.step()

    return {
        "l2_loss": l2_loss(predicted_progress, progress).item(),
        "l1_loss": l1_loss(predicted_progress, progress).item(),
        "count": B,
    }


def main():
    args = parse_args()
    pwd = os.getcwd()
    if 'nfs' in os.getcwd():
        root = '/tudelft.net/staff-umbrella/StudentsCVlab/fransdeboer/'
    else:
        root = '/home/frans/Datasets'

    data_root = os.path.join(root, args.dataset)
    experiment_path = None
    if args.experiment_name and args.experiment_name.lower() != 'none':
        experiment_path = os.path.join(root, "experiments", args.experiment_name)

    if not args.wandb_disable and not args.print_only:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_name,
            tags=args.wandb_tags,
            config={
                'seed': args.seed,
                'experiment_name': args.experiment_name,
                'dataset': args.dataset,
                'data_dir': args.data_dir,
                'flat': args.flat,
                'bboxes': args.bboxes,
                'subsample': args.subsample,
                'rsd_type': args.rsd_type,
                'fps': args.fps,
                'train_split': args.train_split,
                'test_split': args.test_split,
                'batch_size': args.batch_size,
                'iterations': args.iterations,
                'network': args.network,
                'backbone': args.backbone,
                'load_backbone': args.load_backbone,
                'feature_dim': args.feature_dim,
                'embed_dim': args.embed_dim,
                'dropout_chance': args.dropout_chance,
                'pooling_layers': args.pooling_layers,
                'roi_size': args.roi_size,
                'load_experiment': args.load_experiment,
                'load_iteration': args.load_iteration,
                'optimizer': args.optimizer,
                'loss': args.loss,
                'momentum': args.momentum,
                'betas': (args.beta1, args.beta2),
                'lr': args.lr,
                'weight_decay': args.weight_decay,
                'lr_decay': args.lr_decay,
                'lr_decay_every': args.lr_decay_every,
            },
        )

    # TODO: Subsampling
    if "features" in args.data_dir:
        trainset = FeatureDataset(
            data_root, args.data_dir, args.train_split, flat=args.flat
        )
        testset = FeatureDataset(
            data_root, args.data_dir, args.test_split, flat=args.flat
        )
    elif "images" in args.data_dir:
        transform = [transforms.ToTensor()]
        if "tudelft" in root:
            # antialias not available on compute cluster
            transform.append(transforms.Resize((224, 224)))
        else:
            transform.append(transforms.Resize((224, 224), antialias=True))
        transform = transforms.Compose(transform)

        if "ucf24" in args.dataset:
            trainset = UCFDataset(
                data_root,
                args.data_dir,
                args.train_split,
                args.bboxes,
                flat=args.flat,
                transform=transform,
            )
            testset = UCFDataset(
                data_root,
                args.data_dir,
                args.test_split,
                args.bboxes,
                flat=args.flat,
                transform=transform,
            )
        else:
            trainset = ImageDataset(
                data_root,
                args.data_dir,
                args.train_split,
                flat=args.flat,
                transform=transform,
            )
            testset = ImageDataset(
                data_root,
                args.data_dir,
                args.test_split,
                flat=args.flat,
                transform=transform,
            )

    trainloader = DataLoader(
        trainset, batch_size=args.batch_size, num_workers=4, shuffle=True
    )
    testloader = DataLoader(
        testset, batch_size=args.batch_size, num_workers=4, shuffle=False
    )

    # TODO: Sequential RSDNet and ProgressNet
    if args.load_backbone:
        backbone_path = os.path.join(data_root, "train_data", args.load_backbone)
    else:
        backbone_path = None
    if args.network == "progressnet" and args.flat:
        network = ProgressNetFlat(
            args.pooling_layers,
            args.roi_size,
            args.dropout_chance,
            args.embed_dim,
            args.backbone,
            backbone_path,
        )
    elif args.network == "progressnet" and not args.flat:
        raise NotImplementedError()
    elif args.network == "rsdnet" and args.flat:
        network = RSDNetFlat(args.backbone, backbone_path)
    elif args.network == "rsdnet" and not args.flat:
        raise NotImplementedError()
    elif args.network == "ute" and args.flat:
        network = Linear(args.feature_dim, args.embed_dim)
    else:
        raise Exception(
            f"No network for combination {args.network} and flat={args.flat}"
        )

    if args.load_experiment and args.load_iteration:
        network_path = os.path.join(
            root,
            "experiments",
            args.load_experiment,
            f"model_{args.load_iteration}",
        )
        network.load_state_dict(torch.load(network_path))

    if args.optimizer == "sgd":
        optimizer = optim.SGD(
            network.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
    elif args.optimizer == "adam":
        optimizer = optim.Adam(
            network.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
            betas=(args.beta1, args.beta2),
        )
    else:
        raise Exception(f"Optimizer {args.optimizer} does not exist")

    scheduler = optim.lr_scheduler.StepLR(optimizer, args.lr_decay_every, args.lr_decay)

    if args.loss == "l2":
        criterion = nn.MSELoss()
    elif args.loss == "l1":
        criterion = nn.L1Loss()
    elif args.loss == "smooth_l1":
        criterion = nn.SmoothL1Loss()
    else:
        raise Exception(f"Loss {args.loss} does not exist")

    if "features" in args.data_dir and args.flat:
        train_fn = train_flat_features
    elif args.bboxes and args.flat:
        train_fn = train_flat_bbox_frames
    elif "images" in args.data_dir and args.flat:
        train_fn = train_flat_frames
    else:
        raise Exception(
            f"No train function for combination {args.data_dir} and flat={args.flat}"
        )

    experiment = Experiment(
        network,
        criterion,
        optimizer,
        scheduler,
        trainloader,
        testloader,
        train_fn,
        experiment_path,
        args.seed,
        {"l1_loss": 0.0, "l2_loss": 0.0, "count": 0},
    )
    experiment.print()
    if not args.print_only:
        experiment.run(args.iterations, args.log_every, args.test_every)


if __name__ == "__main__":
    main()
