import torch.nn as nn
import argparse
import torch
import os

from .dumb_networks import StaticNet, RandomNet
from .networks import ProgressNet


def get_network(args: argparse.Namespace, device: torch.device) -> nn.Module:
    # get network
    if args.network == 'progressnet':
        network = ProgressNet(args, device)
    elif args.network == 'dumb_static':
        network = StaticNet(device)
    elif args.network == 'dumb_random':
        network = RandomNet(device)
    # load weights from previous experiment
    if args.load_experiment and args.load_iteration:
        path = os.path.join(args.data_root, 'experiments', args.load_experiment, 'models', f'model_{args.load_iteration}.pth')
        network.load_state_dict(torch.load(path))

    return network.to(device)
