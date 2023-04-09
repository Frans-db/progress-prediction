import argparse
import wandb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import random
import numpy as np
import os

from datasets import ProgressDataset


def get_device(device: str) -> torch.device:
    if torch.cuda.is_available() and device == 'cuda':
        return torch.device('cuda')
    return torch.device('cpu')


def init_weights(m: nn.Module) -> None:
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.uniform_(m.bias, a=0, b=0)
    elif isinstance(m, nn.LSTM):
        nn.init.xavier_uniform_(m.weight_hh_l0)
        nn.init.xavier_uniform_(m.weight_ih_l0)
        nn.init.uniform_(m.bias_ih_l0, a=0, b=0)
        nn.init.uniform_(m.bias_hh_l0, a=0, b=0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    # experiment
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--network', type=str, default='sequential_lstm')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--data_root', type=str,
                        default='/home/frans/Datasets/')
    # wandb config
    parser.add_argument('--no_wandb', action='store_true',
                        help='Disable Weights & Biases')
    parser.add_argument('--wandb_group', type=str, default='default',
                        help='Uses to group together results in Weights & Biases')
    # datasets
    parser.add_argument('--train_set', type=str, default='toy')
    parser.add_argument('--test_set', type=str, default='toy')
    parser.add_argument('--data_type', type=str, default='pooled/small')
    # training
    parser.add_argument('--iterations', type=int, default=10_000)
    parser.add_argument('--learning_rate', type=float, default=3e-3)
    parser.add_argument('--losses', nargs='+', default='progress')
    parser.add_argument('--augmentations', nargs='+', default='')
    # forecasting
    parser.add_argument('--delta_t', type=int, default=10)

    return parser.parse_args()


def get_network(network: str) -> nn.Module:
    return None


def init(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if not args.no_wandb:
        wandb.init(
            project='mscfransdeboer',
            config={
                'seed': args.seed,
                'network': args.network,
                'group': args.wandb_group,
                'train_set': args.train_set,
                'test_set': args.test_set,
                'data_type': args.data_type,
                'iterations': args.iterations,
                'learning_rate': args.learning_rate,
                'losses': args.losses,
                'augmentations': args.augmentations,
                'delta_t': args.delta_t,
            }
        )

    train_root = os.path.join(args.data_root, args.train_set)
    test_root = os.path.join(args.data_root, args.test_set)
    # TODO: Sample transform
    trainset = ProgressDataset(
        train_root, args.data_type, 'splitfiles/trainlist01.txt')
    testset = ProgressDataset(
        test_root, args.data_type, 'splitfiles/testlist01.txt')
    trainloader = DataLoader(trainset, batch_size=1,
                             num_workers=4, shuffle=True)
    testloader = DataLoader(testset, batch_size=1,
                            num_workers=4, shuffle=False)

    progressnet = get_network(args.network).to(device)
    progressnet.apply(init_weights)


def main() -> None:
    args = parse_args()
    device = get_device(args.device)
    init(args)


if __name__ == '__main__':
    main()
