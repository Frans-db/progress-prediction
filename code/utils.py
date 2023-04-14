import argparse
import torch

def parse_args() -> argparse.Namespace:
    networks = ['progressnet']

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
    parser.add_argument('--network', type=str, choices=networks)
    # dataset
    parser.add_argument('--train_set', type=str, default='breakfast')
    parser.add_argument('--test_set', type=str, default='breakfast')
    parser.add_argument('--train_split', type=str, default='train_s1.txt')
    parser.add_argument('--test_split', type=str, default='test_s1.txt')
    parser.add_argument('--data_type', type=str, default='features/dense_trajectories')
    parser.add_argument('--data_mode', type=str, default='sequential', choices=['sequential', 'individual'])
    parser.add_argument('--data_modifier', type=str, default=None, choices=['random', 'ones', 'indices'])
    # training
    parser.add_argument('--iterations', type=int, default=1500)
    parser.add_argument('--lr', type=float, default=3e-3)
    parser.add_argument('--lr_decay_every', type=int, default=500)
    parser.add_argument('--lr_decay', type=float, default=1/2)
    parser.add_argument('--augmentations', nargs='+', default='', choices=['subsection', 'subsample'])
    # testing
    parser.add_argument('--test_every', type=int, default=250)

    return parser.parse_args()

def get_device(device: str) -> torch.device:
    if torch.cuda.is_available() and device == 'cuda':
        return torch.device('cuda')
    return torch.device('cpu')