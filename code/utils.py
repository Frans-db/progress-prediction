import argparse
import torch
import wandb
import random
import numpy as np

def parse_args() -> argparse.Namespace:
    networks = ['progressnet', 'progressnet_features', 'progressnet_boundingboxes']

    parser = argparse.ArgumentParser()
    # experiment
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--experiment_name', type=str, default=None)
    parser.add_argument('--data_root', type=str, default='/home/frans/Datasets/')
    # wandb
    parser.add_argument('--wandb_disable', action='store_true')
    parser.add_argument('--wandb_project', type=str, default='mscfransdeboer')
    parser.add_argument('--wandb_group', type=str, default=None)
    parser.add_argument('--wandb_tags', nargs='+', default=None)
    # network
    parser.add_argument('--network', type=str, default='progressnet', choices=networks)
    parser.add_argument('--embedding_size', type=int, default=64)
    parser.add_argument('--initialisation', type=str, default='xavier', choices=['random', 'xavier'])
    parser.add_argument('--dropout_chance', type=float, default=0.0)
    # dataset
    parser.add_argument('--train_set', type=str, default='breakfast')
    parser.add_argument('--test_set', type=str, default='breakfast')
    parser.add_argument('--train_split', type=str, default='test_s1_small.txt')
    parser.add_argument('--test_split', type=str, default='test_s1_small.txt')
    parser.add_argument('--data_type', type=str, default='features/dense_trajectories')
    parser.add_argument('--data_modifier', type=str, default=None, choices=['indices', 'ones', 'randoms'])
    parser.add_argument('--bounding_boxes', action='store_true')
    # training
    parser.add_argument('--iterations', type=int, default=1500)
    parser.add_argument('--lr', type=float, default=0.001)
    # TODO (maybe?): Adam betas, weight_decay
    parser.add_argument('--lr_decay_every', type=int, default=500)
    parser.add_argument('--lr_decay', type=float, default=1/2)
    parser.add_argument('--subsection_chance', type=float, default=0.0)
    parser.add_argument('--subsample_chance', type=float, default=0.0)
    # testing
    parser.add_argument('--test_every', type=int, default=250)

    return parser.parse_args()

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
        'initialisation': args.initialisation,
        'dropout_chance': args.dropout_chance,
        # dataset
        'train_set': args.train_set,
        'test_set': args.test_set,
        'train_split': args.train_split,
        'test_split': args.test_split,
        'data_type': args.data_type,
        'data_modifier': args.data_modifier,
        'bounding_boxes': args.bounding_boxes,
        # training
        'iterations': args.iterations,
        'lr': args.lr,
        'lr_decay_every': args.lr_decay_every,
        'lr_decay': args.lr_decay,
        'subsection_chance': args.subsection_chance,
        'subsample_chance': args.subsample_chance,
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