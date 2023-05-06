from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
import argparse
import wandb
import torch
from tqdm import tqdm
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
    parser.add_argument("--wandb_group", type=str, default=None)
    parser.add_argument("--wandb_disable", action="store_true")
    # data
    parser.add_argument("--dataset", type=str, default="cholec80")
    parser.add_argument("--data_dir", type=str, default="features/i3d_embeddings")
    parser.add_argument("--flat", action="store_true")
    parser.add_argument("--bboxes", action="store_true")
    parser.add_argument("--indices", action="store_true")
    parser.add_argument("--indices_normalizer", type=float, default=1.0)
    parser.add_argument("--subsample", action="store_true")
    parser.add_argument(
        "--rsd_type", type=str, default="none", choices=["none", "minutes", "seconds"]
    )
    parser.add_argument("--rsd_normalizer", type=float, default=1.0)
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--train_split", type=str, default="t12_0.txt")
    parser.add_argument("--test_split", type=str, default="v_0.txt")
    # training
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--iterations", type=int, default=503 * 60)
    # network
    parser.add_argument(
        "--network",
        type=str,
        default="progressnet",
        choices=["ute", "progressnet", "rsdnet"],
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
    parser.add_argument("--embed", action="store_true")
    parser.add_argument("--embed_batch_size", type=int, default=10)
    parser.add_argument("--embed_dir", type=str, default=None)
    parser.add_argument("--eval", action="store_true")

    parser.add_argument("--num_workers", type=int, default=4)

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
    progress = batch[-1]
    data = batch[1:-1]
    data = tuple([d.to(device) for d in data])

    B = data[0].shape[0]
    predicted_progress = network(*data).reshape(B)
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

def train_rsd(network, criterion, batch, device, optimizer=None):
    l2_loss = nn.MSELoss(reduction="sum")
    l1_loss = nn.MSELoss(reduction="sum")
    smooth_l1_loss = nn.MSELoss(reduction='sum')

    rsd = batch[-2] / network.rsd_normalizer
    progress = batch[-1]

    data = batch[1:-2]
    data = tuple([d.to(device) for d in data])
    S = data[0].shape[1]

    predicted_rsd, predicted_progress = network(*data)

    rsd = rsd.to(device)
    progress = progress.to(device)
    if optimizer:
        optimizer.zero_grad()
        (criterion(predicted_rsd, rsd) + criterion(predicted_progress, progress)).backward()
        optimizer.step()

    return {
            "rsd_l1_loss": l1_loss(predicted_rsd, rsd), 
            'rsd_smooth_l1_loss': smooth_l1_loss(predicted_rsd, rsd),
            "rsd_l2_loss": l2_loss(predicted_rsd, rsd),
            'rsd_normal_l1_loss': l1_loss(predicted_rsd * network.rsd_normalizer, rsd * network.rsd_normalizer),

            "progress_l1_loss": l1_loss(predicted_progress, progress), 
            'progress_smooth_l1_loss': smooth_l1_loss(predicted_progress, progress),
            "progress_l2_loss": l2_loss(predicted_progress, progress),

            "count": S,
    }


def embed_frames(network, batch, device, batch_size: int):
    data = batch[1:-1]
    data = tuple([torch.split(d.squeeze(dim=0), batch_size) for d in data])

    embeddings = []
    for samples in zip(*data):
        samples = tuple([sample.to(device) for sample in samples])
        sample_embeddings = network.embed(*samples)
        embeddings.extend(sample_embeddings.tolist())

    return batch[0][0], embeddings


def main():
    args = parse_args()
    if "nfs" in os.getcwd():
        root = "/tudelft.net/staff-umbrella/StudentsCVlab/fransdeboer/"
    else:
        root = "/home/frans/Datasets"

    data_root = os.path.join(root, args.dataset)
    experiment_path = None
    if args.experiment_name and args.experiment_name.lower() != "none":
        experiment_path = os.path.join(root, "experiments", args.experiment_name)

    if (
        not args.wandb_disable
        and not args.print_only
        and not args.eval
        and not args.embed
    ):
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_name,
            group=args.wandb_group,
            tags=args.wandb_tags,
            config={
                "seed": args.seed,
                "experiment_name": args.experiment_name,
                "dataset": args.dataset,
                "data_dir": args.data_dir,
                "flat": args.flat,
                "indices": args.indices,
                "indices_normalizer": args.indices_normalizer,
                "bboxes": args.bboxes,
                "subsample": args.subsample,
                "rsd_type": args.rsd_type,
                "rsd_normalizer": args.rsd_normalizer,
                "fps": args.fps,
                "train_split": args.train_split,
                "test_split": args.test_split,
                "batch_size": args.batch_size,
                "iterations": args.iterations,
                "network": args.network,
                "backbone": args.backbone,
                "load_backbone": args.load_backbone,
                "feature_dim": args.feature_dim,
                "embed_dim": args.embed_dim,
                "dropout_chance": args.dropout_chance,
                "pooling_layers": args.pooling_layers,
                "roi_size": args.roi_size,
                "load_experiment": args.load_experiment,
                "load_iteration": args.load_iteration,
                "optimizer": args.optimizer,
                "loss": args.loss,
                "momentum": args.momentum,
                "betas": (args.beta1, args.beta2),
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "lr_decay": args.lr_decay,
                "lr_decay_every": args.lr_decay_every,
            },
        )

    # TODO: Subsampling
    if "features" in args.data_dir:
        trainset = FeatureDataset(
            data_root,
            args.data_dir,
            args.train_split,
            args.flat,
            args.indices,
            args.indices_normalizer,
            args.rsd_type,
            args.fps,
        )
        testset = FeatureDataset(
            data_root,
            args.data_dir,
            args.test_split,
            args.flat,
            args.indices,
            args.indices_normalizer,
            args.rsd_type,
            args.fps,
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
                args.flat,
                args.indices,
                transform=transform,
            )
            testset = UCFDataset(
                data_root,
                args.data_dir,
                args.test_split,
                args.bboxes,
                args.flat,
                args.indices,
                transform=transform,
            )
        else:
            trainset = ImageDataset(
                data_root,
                args.data_dir,
                args.train_split,
                args.flat,
                args.indices,
                transform=transform,
            )
            testset = ImageDataset(
                data_root,
                args.data_dir,
                args.test_split,
                args.flat,
                args.indices,
                transform=transform,
            )

    trainloader = DataLoader(
        trainset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True
    )
    testloader = DataLoader(
        testset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False
    )

    # TODO: Sequential RSDNet and ProgressNet
    if args.load_backbone:
        backbone_path = os.path.join(data_root, "train_data", args.load_backbone)
    else:
        backbone_path = None
    if args.network == "progressnet" and (args.flat or args.embed):
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
    elif args.network == "rsdnet" and (args.flat or args.embed):
        network = RSDNetFlat(args.backbone, backbone_path)
    elif args.network == "rsdnet" and not args.flat:
        network = RSDNet(args.feature_dim, args.rsd_normalizer, args.dropout_chance)
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
            f"model_{args.load_iteration}.pth",
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

    result = {"l1_loss": 0.0, "l2_loss": 0.0, "count": 0}
    if args.embed:
        train_fn = None
    elif "features" in args.data_dir and args.flat:
        train_fn = train_flat_features
    elif 'features' in args.data_dir and args.rsd_type != 'none' and not args.flat:
        train_fn = train_rsd
        result = {
            "rsd_l1_loss": 0.0, 
            'rsd_smooth_l1_loss': 0.0,
            "rsd_l2_loss": 0.0, 
            'rsd_normal_l1_loss': 0.0,

            "progress_l1_loss": 0.0, 
            'progress_smooth_l1_loss': 0.0,
            "progress_l2_loss": 0.0, 

            "count": 0
        }
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
    if args.eval:
        experiment.eval()
    elif args.embed:
        if args.flat:
            raise Exception("Can't embed flat dataset")
        network.eval()
        with torch.no_grad():
            save_dir = os.path.join(data_root, args.embed_dir)
            os.mkdir(save_dir)
            for batch in tqdm(trainloader):
                video_name, embeddings = embed_frames(
                    network, batch, experiment.device, args.embed_batch_size
                )
                txt = []
                for embedding in embeddings:
                    txt.append(" ".join(map(str, embedding)))
                save_path = os.path.join(save_dir, f'{video_name}.txt')
                with open(save_path, "w+") as f:
                    f.write("\n".join(txt))

    elif not args.print_only:
        experiment.run(args.iterations, args.log_every, args.test_every)


if __name__ == "__main__":
    main()
