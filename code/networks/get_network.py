import torch.nn as nn
import argparse
import torch
import os

from .dumb_networks import StaticNet, RandomNet
from .networks import ProgressNet, ProgressNetPooling, ProgressNetFeatures, ProgressResNet

def init_weights(m: nn.Module) -> None:
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.uniform_(m.bias, a=0, b=0)
    elif isinstance(m, nn.LSTM):
        nn.init.xavier_uniform_(m.weight_hh_l0)
        nn.init.xavier_uniform_(m.weight_ih_l0)
        nn.init.uniform_(m.bias_ih_l0, a=0, b=0)
        nn.init.uniform_(m.bias_hh_l0, a=0, b=0)


def get_network(args: argparse.Namespace, device: torch.device) -> nn.Module:
    if args.network == 'progressnet':
        network = ProgressNet(args, device)
    elif args.network == 'progressnet_pooling':
        network = ProgressNetPooling(args, device)
    elif args.network == 'progressnet_features':
        network = ProgressNetFeatures(args, device)
    elif args.network == 'progressnet_resnet':
        network = ProgressResNet(args, device)
    elif args.network == 'dumb_static':
        network = StaticNet(device)
    elif args.network == 'dumb_random':
        network = RandomNet(device)

    if args.load_experiment and args.load_iteration:
        path = os.path.join(args.data_root, 'experiments', args.load_experiment, 'models', f'model_{args.load_iteration}.pth')
        network.load_state_dict(torch.load(path))
    elif args.initialisation == 'xavier':
        network.apply(init_weights)
    return network.to(device)
