import argparse
import torch
import wandb
import random
import numpy as np
import os
import json

def parse_args(parse=True) -> argparse.Namespace:
    networks = ['progressnet', 
                'rsdnet', 'rsdflat',
                'dumb_static', 'dumb_random']

    parser = argparse.ArgumentParser()
    # experiment
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--data_root', type=str, default='/home/frans/Datasets/')
    parser.add_argument('--experiment_name', type=str, default=None)
    # wandb
    parser.add_argument('--wandb_disable', action='store_true')
    parser.add_argument('--wandb_watch', action='store_true')
    parser.add_argument('--wandb_project', type=str, default='mscfransdeboer')
    parser.add_argument('--wandb_name', type=str, default=None)
    parser.add_argument('--wandb_group', type=str, default=None)
    parser.add_argument('--wandb_tags', nargs='+', default=None)
    # network
    parser.add_argument('--network', type=str, default='progressnet', choices=networks)
    parser.add_argument('--embedding_size', type=int, default=1024)
    parser.add_argument('--pooling_layers', type=int, nargs='+', default=[1, 2, 3])
    parser.add_argument('--roi_type', type=str, default='align', choices=['pool', 'align'])
    parser.add_argument('--roi_size', type=int, default=1)
    parser.add_argument('--roi_scale', type=float, default=1.0)
    parser.add_argument('--backbone', type=str, default='vgg512')
    parser.add_argument('--backbone_name', type=str, default=None)
    parser.add_argument('--backbone_depth', type=int, default=34)
    parser.add_argument('--backbone_gradients', action='store_true')
    parser.add_argument('--backbone_channels', type=int, default=512)
    parser.add_argument('--dropout_chance', type=float, default=0.5)
    parser.add_argument('--finetune', action='store_true')
    # network loading
    parser.add_argument('--load_experiment', type=str, default=None)
    parser.add_argument('--load_iteration', type=int, default=None)
    # dataset
    parser.add_argument('--dataset', type=str, default='ucf24')
    parser.add_argument('--dataset_type', type=str, default='boundingboxes', choices=['boundingboxes', 'images'])
    parser.add_argument('--data_type', type=str, default='rgb-images')
    parser.add_argument('--max_length', type=int, default=400)
    parser.add_argument('--train_split', type=str, default='train_telic.txt')
    parser.add_argument('--test_split', type=str, default='test_telic.txt')
    parser.add_argument('--data_modifier', type=str, default=None, choices=['indices', 'ones', 'randoms'])
    # bf 2113.340410958904
    # ucf24 173.42794759825327
    parser.add_argument('--data_modifier_value', type=float, default=1.0)
    parser.add_argument('--resize', type=int, default=300)
    parser.add_argument('--antialias', action='store_true')
    # training
    parser.add_argument('--iterations', type=int, default=25000)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--lr_decay_every', type=int, default=1_000_000)
    parser.add_argument('--lr_decay', type=float, default=1)
    parser.add_argument('--subsection_chance', type=float, default=1.0)
    parser.add_argument('--subsample_chance', type=float, default=1.0)
    # logging
    parser.add_argument('--test_every', type=int, default=1000)
    parser.add_argument('--log_every', type=int, default=250)
    # misc
    parser.add_argument('--debug', action='store_true')
    

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
        'experiment': {
            'seed': args.seed,
            'experiment_name': args.experiment_name,
        },
        'network': {
            'network': args.network,
            'embedding_size': args.embedding_size,
            'pooling_layers': args.pooling_layers,
            'roi': {
                'roi_type': args.roi_type,   
                'roi_size': args.roi_size,
                'roi_scale': args.roi_scale
            },
            'backbone': {
                'backbone': args.backbone,
                'backbone_name': args.backbone_name,
                'backbone_depth': args.backbone_depth,
                'backbone_gradients': args.backbone_gradients,
                'backbone_channels': args.backbone_channels,
            },
            'loading': {
                'load_experiment': args.load_experiment,
                'load_iteration': args.load_iteration,
            },
            'dropout_chance': args.dropout_chance,
            'finetune': args.finetune,
        },
        'data': {
            'dataset': args.dataset,
            'dataset_tyoe': args.dataset_type,
            'data_type': args.data_type,
            'max_length': args.max_length,
            'splits': {
                'train_split': args.train_split,
                'test_split': args.test_split,
            },
            'modifiers': {
                'data_modifier': args.data_modifier,
                'data_modifier_value': args.data_modifier_value,
            },
            'resize': {
                'resize': args.resize,
                'antialias': args.antialias,
            },
        },
        'training': {
            'iterations': args.iterations,
            'batch_size': args.batch_size,
            'optimizer': {
                'lr': args.lr,
                'betas': (args.beta1, args.beta2),
                'weight_decay': args.weight_decay,
            },
            'scheduler': {
                'lr_decay_every': args.lr_decay_every,
                'lr_decay': args.lr_decay,
            },
            'sampling': {
                'subsection_chance': args.subsection_chance,
                'subsample_chance': args.subsample_chance,
            },

        },
        'logging': {
            'test_every': args.test_every,
            'log_every': args.log_every,
        },
        
        'debug': args.debug,
    }

    if args.experiment_name:
        experiment_root = os.path.join(args.data_root, 'experiments', args.experiment_name)
        os.mkdir(experiment_root)
        os.mkdir(os.path.join(experiment_root, 'results'))
        os.mkdir(os.path.join(experiment_root, 'models'))
        os.mkdir(os.path.join(experiment_root, 'plots'))
        with open(os.path.join(experiment_root, 'config.json'), 'w+') as f:
            json.dump(config, f)


    if not args.wandb_disable and not args.debug:
        wandb.init(
            project=args.wandb_project,
            group=args.wandb_group,
            tags=args.wandb_tags,
            name=args.wandb_name,
            config=config
        )