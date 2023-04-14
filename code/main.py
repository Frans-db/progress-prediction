import random
import torch
import numpy as np
import argparse
import wandb
import json
import os

from utils import parse_args, get_device
from datasets import get_datasets


def init(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    config = {
        # experiment
        'seed': args.seed,
        'experiment_name': args.experiment_name,
        # network
        'network': args.network,
        # dataset
        'train_set': args.train_set,
        'test_set': args.test_set,
        'train_split': args.train_split,
        'test_split': args.test_split,
        'data_type': args.data_type,
        # training
        'iterations': args.iterations,
        'lr': args.lr,
        'lr_decay_every': args.lr_decay_every,
        'lr_decay': args.lr_decay,
        'augmentations': args.augmentations,
        'data_modifier': args.data_modifier,
        # testing
        'test_every': args.test_every,
    }

    if not args.wandb_disable:
        wandb.init(
            project=args.wandb_project,
            group=args.wandb_group,
            tags=args.wandb_tags,
            config=config
        )
    if args.experiment_name:
        experiment_root = os.path.join(
            args.data_root, 'experiments', args.experiment_name
        )
        os.mkdir(experiment_root)
        os.mkdir(os.path.join(experiment_root, 'results'))
        with open(os.path.join(experiment_root, 'config.json'), 'w+') as f:
            json.dump(config, f)


def main() -> None:
    args = parse_args()
    device = get_device(args.device)
    init(args)

    train_set, test_set = get_datasets(args)
    # get train & test set (function)
    # get (data) augmentations
    # get sample augmentations
    # get network (function)
    # init network
    # for parameters: xavier / random
    # for static: depending on train set
    # get optimizer & scheduler

    # training


if __name__ == '__main__':
    main()
