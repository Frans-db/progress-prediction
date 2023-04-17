import argparse
import torch
import wandb
import random
import numpy as np

def parse_args() -> argparse.Namespace:
    networks = ['progressnet', 'progressnet_features', 'progressnet_boundingboxes', 
                'progressnet_categories', 'progressnet_features_2d', 'progressnet_boundingboxes_2d',
                'progressnet_resnet', 'dumb_static', 'dumb_random', 'progressnet_boundingboxes_vgg']

    parser = argparse.ArgumentParser()
    # experiment
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--data_root', type=str, default='/home/frans/Datasets/')
    # wandb
    parser.add_argument('--wandb_disable', action='store_true')
    parser.add_argument('--wandb_project', type=str, default='mscfransdeboer')
    parser.add_argument('--wandb_name', type=str, default=None)
    parser.add_argument('--wandb_group', type=str, default=None)
    parser.add_argument('--wandb_tags', nargs='+', default=None)
    # network
    parser.add_argument('--network', type=str, default='progressnet_features', choices=networks)
    parser.add_argument('--embedding_size', type=int, default=64)
    parser.add_argument('--initialisation', type=str, default='xavier', choices=['random', 'xavier'])
    parser.add_argument('--dropout_chance', type=float, default=0.0)
    # dataset
    parser.add_argument('--train_set', type=str, default='breakfast')
    parser.add_argument('--test_set', type=str, default='breakfast')
    parser.add_argument('--train_split', type=str, default='train_s1.txt')
    parser.add_argument('--test_split', type=str, default='test_s1.txt')
    parser.add_argument('--data_type', type=str, default='features/dense_trajectories')
    parser.add_argument('--category_directory', type=str, default=None)
    parser.add_argument('--num_categories', type=int, default=48)
    parser.add_argument('--data_modifier', type=str, default=None, choices=['indices', 'ones', 'randoms'])
    # bf 2113.340410958904
    # ucf24 173.42794759825327
    parser.add_argument('--data_modifier_value', type=float, default=1.0)
    parser.add_argument('--bounding_boxes', action='store_true')
    # training
    parser.add_argument('--iterations', type=int, default=25000)
    parser.add_argument('--loss', type=str, default='l2', choices=['l1', 'l2'])
    parser.add_argument('--average_loss', action='store_true')
    parser.add_argument('--bo', action='store_true')
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
    parser.add_argument('--testing_fps', type=int, default=1)

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
        'category_directory': args.category_directory,
        'num_categories': args.num_categories,
        'data_modifier': args.data_modifier,
        'data_modifier_value': args.data_modifier_value,
        'bounding_boxes': args.bounding_boxes,
        # training
        'iterations': args.iterations,
        'loss': args.loss,
        'average_loss': args.average_loss,
        'bo': args.bo,
        'lr': args.lr,
        'betas': (args.beta1, args.beta2),
        'weight_decay': args.weight_decay,
        'lr_decay_every': args.lr_decay_every,
        'lr_decay': args.lr_decay,
        'subsection_chance': args.subsection_chance,
        'subsample_chance': args.subsample_chance,
        # testing
        'test_every': args.test_every,
        'testing_fps': args.testing_fps
    }

    if not args.wandb_disable:
        wandb.init(
            project=args.wandb_project,
            group=args.wandb_group,
            tags=args.wandb_tags,
            name=args.wandb_name,
            config=config
        )