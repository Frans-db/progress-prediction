import argparse
import torch
import wandb
import random
import numpy as np
import os
import json

def parse_args(parse=True) -> argparse.Namespace:
    networks = ['progressnet', 'dumb_static', 'dumb_random']

    parser = argparse.ArgumentParser()
    # experiment
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--data_root', type=str, default='/home/frans/Datasets/')
    parser.add_argument('--experiment_name', type=str, default=None)
    # wandb
    parser.add_argument('--wandb_disable', action='store_true')
    parser.add_argument('--wandb_project', type=str, default='mscfransdeboer')
    parser.add_argument('--wandb_name', type=str, default=None)
    parser.add_argument('--wandb_group', type=str, default=None)
    parser.add_argument('--wandb_tags', nargs='+', default=None)
    # network
    parser.add_argument('--network', type=str, default='progressnet', choices=networks)
    parser.add_argument('--embedding_size', type=int, default=4096)
    parser.add_argument('--pooling_layers', type=int, nargs='+', default=[1, 3, 5])
    parser.add_argument('--roi_size', type=int, default=1)
    parser.add_argument('--initialisation', type=str, default='xavier', choices=['random', 'xavier'])
    parser.add_argument('--dropout_chance', type=float, default=0.5)
    parser.add_argument('--basemodel', type=str, default='vgg512', choices=['vgg512', 'vgg1024'])
    parser.add_argument('--basemodel_name', type=str, default='vgg_512.pth')
    parser.add_argument('--basemodel_gradients', action='store_true')
    parser.add_argument('--finetune', action='store_true')
    # dataset
    parser.add_argument('--dataset', type=str, default='ucf24')
    parser.add_argument('--train_split', type=str, default='train.txt')
    parser.add_argument('--test_split', type=str, default='test.txt')
    parser.add_argument('--data_type', type=str, default='rgb-images')
    parser.add_argument('--data_modifier', type=str, default=None, choices=['indices', 'ones', 'randoms'])
    parser.add_argument('--data_modifier_value', type=float, default=1.0)
    parser.add_argument('--bounding_boxes', action='store_true')
    # training
    parser.add_argument('--iterations', type=int, default=25000)
    parser.add_argument('--loss', type=str, default='l2', choices=['l1', 'l2'])
    parser.add_argument('--lr', type=float, default=3e-3)
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--lr_decay_every', type=int, default=1_000_000)
    parser.add_argument('--lr_decay', type=float, default=1)
    parser.add_argument('--subsection_chance', type=float, default=1.0)
    parser.add_argument('--subsample_chance', type=float, default=1.0)
    # testing
    parser.add_argument('--test_every', type=int, default=1000)

    if parse:
        return parser.parse_args()
    return parser

def get_device(device: str) -> torch.device:
    if torch.cuda.is_available() and device == 'cuda':
        return torch.device('cuda')
    return torch.device('cpu')

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
        'embedding_size': args.embedding_size,
        'pooling_layers': args.pooling_layers,
        'roi_size': args.roi_size,
        'initialisation': args.initialisation,
        'dropout_chance': args.dropout_chance,
        'basemodel': args.basemodel,
        'basemodel_name': args.basemodel_name,
        'basemodel_gradients': args.basemodel_gradients,
        'finetune': args.finetune,
        # dataset
        'dataset': args.dataset,
        'train_split': args.train_split,
        'test_split': args.test_split,
        'data_type': args.data_type,
        'data_modifier': args.data_modifier,
        'data_modifier_value': args.data_modifier_value,
        'bounding_boxes': args.bounding_boxes,
        # training
        'iterations': args.iterations,
        'loss': args.loss,
        'lr': args.lr,
        'betas': (args.beta1, args.beta2),
        'weight_decay': args.weight_decay,
        'lr_decay_every': args.lr_decay_every,
        'lr_decay': args.lr_decay,
        'subsection_chance': args.subsection_chance,
        'subsample_chance': args.subsample_chance,
        # testing
        'test_every': args.test_every,
    }

    if args.experiment_name:
        experiment_root = os.path.join(args.data_root, 'experiments', args.experiment_name)
        os.mkdir(experiment_root)
        os.mkdir(os.path.join(experiment_root, 'results'))
        os.mkdir(os.path.join(experiment_root, 'models'))
        os.mkdir(os.path.join(experiment_root, 'plots'))
        with open(os.path.join(experiment_root, 'config.json'), 'w+') as f:
            json.dump(config, f)


    if not args.wandb_disable:
        wandb.init(
            project=args.wandb_project,
            group=args.wandb_group,
            tags=args.wandb_tags,
            name=args.wandb_name,
            config=config
        )