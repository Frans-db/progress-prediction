from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
import torch
from tqdm import tqdm
import os

from arguments import parse_args, wandb_init
from networks import Linear
from networks import ProgressNet
from networks import RSDNet, RSDNetFlat, LSTMNet
from networks import ToyNet, ResNet
from datasets import FeatureDataset, ImageDataset, UCFDataset
from datasets import Subsample, Subsection
from experiment import Experiment


def train_flat_features(network, criterion, batch, device, optimizer=None):
    l2_loss = nn.MSELoss(reduction="sum")
    l1_loss = nn.L1Loss(reduction="sum")
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
        "l1_loss": l1_loss(predicted_progress * 100, progress * 100).item(),
        "count": B,
    }


def train_flat_frames(network, criterion, batch, device, optimizer=None):
    l2_loss = nn.MSELoss(reduction="sum")
    l1_loss = nn.L1Loss(reduction="sum")
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
        "l1_loss": l1_loss(predicted_progress * 100, progress * 100).item(),
        "count": B,
    }


def train_progress(network, criterion, batch, device, optimizer=None, return_results=False):
    l2_loss = nn.MSELoss(reduction="sum")
    l1_loss = nn.L1Loss(reduction="sum")
    progress = batch[-1]
    data = batch[1:-1]
    data = tuple([d.to(device) for d in data])

    S = data[0].shape[1]
    predicted_progress = network(*data)
    if return_results:
        return predicted_progress.cpu()
    progress = progress.to(device)
    if optimizer:
        optimizer.zero_grad()
        criterion(predicted_progress, progress).backward()
        optimizer.step()

    return {
        "l2_loss": l2_loss(predicted_progress, progress).item(),
        "l1_loss": l1_loss(predicted_progress * 100, progress * 100).item(),
        "count": S,
    }


def train_rsd(network, criterion, batch, device, optimizer=None):
    l2_loss = nn.MSELoss(reduction="sum")
    l1_loss = nn.L1Loss(reduction="sum")
    smooth_l1_loss = nn.SmoothL1Loss(reduction="sum")

    rsd = batch[-2] / network.rsd_normalizer
    progress = batch[-1]
    S = progress.shape[1]

    data = batch[1:-2]
    data = tuple([d.to(device) for d in data])

    predicted_rsd, predicted_progress = network(*data)

    rsd = rsd.to(device)
    progress = progress.to(device)
    if optimizer:
        optimizer.zero_grad()
        loss = criterion(predicted_rsd, rsd) + criterion(predicted_progress, progress)
        loss.backward()
        optimizer.step()

    return {
        "rsd_l1_loss": l1_loss(predicted_rsd, rsd),
        "rsd_smooth_l1_loss": smooth_l1_loss(predicted_rsd, rsd),
        "rsd_l2_loss": l2_loss(predicted_rsd, rsd),
        "rsd_normal_l1_loss": l1_loss(
            predicted_rsd * network.rsd_normalizer, rsd * network.rsd_normalizer
        ),
        "l1_loss": l1_loss(predicted_progress * 100, progress * 100),
        "smooth_l1_loss": smooth_l1_loss(predicted_progress, progress),
        "l2_loss": l2_loss(predicted_progress, progress),
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

    # root can be set manually, but can also be obtained automatically so wandb sweeps work properly
    if args.root is not None:
        root = args.root
    elif "nfs" in os.getcwd():
        root = "/tudelft.net/staff-umbrella/StudentsCVlab/fransdeboer/"
    else:
        root = "/home/frans/Datasets"
    data_root = os.path.join(root, args.dataset)

    experiment_path = None
    if args.experiment_name and args.experiment_name.lower() != "none":
        experiment_path = os.path.join(root, "experiments", args.experiment_name)

    if args.subsample:
        subsample = transforms.Compose([Subsection(), Subsample()])
    else:
        subsample = None

    # TODO: Combine datasets
    if "images" in args.data_dir:
        transform = [transforms.ToTensor()]
        if not args.no_resize:
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
                args.subsample_fps,
                args.random,
                args.indices,
                args.indices_normalizer,
                args.rsd_type,
                args.fps,
                transform=transform,
                sample_transform=subsample,
            )
            testset = UCFDataset(
                data_root,
                args.data_dir,
                args.test_split,
                args.bboxes,
                args.flat,
                args.subsample_fps,
                args.random,
                args.indices,
                args.indices_normalizer,
                args.rsd_type,
                args.fps,
                transform=transform,
            )
        else:
            trainset = ImageDataset(
                data_root,
                args.data_dir,
                args.train_split,
                args.flat,
                args.subsample_fps,
                args.random,
                args.indices,
                args.indices_normalizer,
                args.shuffle,
                transform=transform,
                sample_transform=subsample,
            )
            testset = ImageDataset(
                data_root,
                args.data_dir,
                args.test_split,
                args.flat,
                args.subsample_fps,
                args.random,
                args.indices,
                args.indices_normalizer,
                args.shuffle,
                transform=transform,
            )
    else:
        trainset = FeatureDataset(
            data_root,
            args.data_dir,
            args.train_split,
            args.flat,
            args.subsample_fps,
            args.random,
            args.indices,
            args.indices_normalizer,
            args.rsd_type,
            args.fps,
            sample_transform=subsample,
        )
        testset = FeatureDataset(
            data_root,
            args.data_dir,
            args.test_split,
            args.flat,
            args.subsample_fps,
            args.random,
            args.indices,
            args.indices_normalizer,
            args.rsd_type,
            args.fps,
        )

    trainloader = DataLoader(
        trainset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True
    )
    testloader = DataLoader(
        testset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False
    )

    # TODO: Reorganise networks
    if args.load_backbone:
        backbone_path = os.path.join(data_root, "train_data", args.load_backbone)
    else:
        backbone_path = None

    if args.network == "progressnet":
        network = ProgressNet(
            args.pooling_layers,
            args.roi_size,
            args.dropout_chance,
            args.embed_dim,
            args.backbone,
            backbone_path,
        )
    elif args.network == "rsdnet_flat":
        network = RSDNetFlat(args.backbone, backbone_path)
    elif args.network == 'lstmnet':
        network = LSTMNet(args.feature_dim, args.dropout_chance)
    elif args.network == "rsdnet":
        network = RSDNet(args.feature_dim, args.rsd_normalizer, args.dropout_chance)
    elif args.network == "ute":
        network = Linear(args.feature_dim, args.embed_dim, args.dropout_chance)
    elif args.network == "toynet":
        network = ToyNet(dropout_chance=args.dropout_chance)
    elif args.network == "resnet":
        network = ResNet(args.backbone, backbone_path, args.dropout_chance)
    else:
        raise Exception(f"Network {args.network} does not exist")

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

    # TODO: Redo
    result = {"l1_loss": 0.0, "l2_loss": 0.0, "count": 0}
    if args.embed:
        train_fn = None
    elif "images" not in args.data_dir and args.flat:
        train_fn = train_flat_features
    elif "images" not in args.data_dir and args.rsd_type != "none" and not args.flat:
        train_fn = train_rsd
        result = {
            "rsd_l1_loss": 0.0,
            "rsd_smooth_l1_loss": 0.0,
            "rsd_l2_loss": 0.0,
            "rsd_normal_l1_loss": 0.0,
            "l1_loss": 0.0,
            "smooth_l1_loss": 0.0,
            "l2_loss": 0.0,
            "count": 0,
        }
    elif "images" in args.data_dir and args.flat:
        train_fn = train_flat_frames
    else:
        train_fn = train_progress

    wandb_init(args)
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
        result,
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
            os.makedirs(save_dir, exist_ok=True)
            for batch in tqdm(trainloader):
                video_name, embeddings = embed_frames(
                    network, batch, experiment.device, args.embed_batch_size
                )
                txt = []
                for embedding in embeddings:
                    txt.append(" ".join(map(str, embedding)))
                save_path = os.path.join(save_dir, f"{video_name}.txt")
                recursive_dir = '/'.join(save_path.split('/')[:-1])
                os.makedirs(recursive_dir, exist_ok=True)
                with open(save_path, "w+") as f:
                    f.write("\n".join(txt))
    elif args.save:
        for i, batch in enumerate(tqdm(testloader)):
            progress = train_progress(network, criterion, batch, experiment.device, return_results=True)
            progress = torch.flatten(progress).tolist()
            txt = '\n'.join(map(str, progress))
            with open(f'./data/{i}.txt', 'w+') as f:
                f.write(txt)
    elif not args.print_only:
        experiment.run(args.iterations, args.log_every, args.test_every)


if __name__ == "__main__":
    main()
