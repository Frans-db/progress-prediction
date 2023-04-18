import torch.nn as nn
import argparse
import torch

from .dumb_networks import StaticNet, RandomNet
from .networks import ProgressNet, ProgressNetFeatures, ProgressNetBoundingBoxes, ProgressNetCategories, ProgressNetBoundingBoxes2D, ProgressNetFeatures2D
from .networks import ProgressResNet, ProgressNetBoundingBoxesVGG, ProgressResNetIndices

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
        network = ProgressNet(args.embedding_size, args.dropout_chance)
    elif args.network == 'progressnet_features':
        network = ProgressNetFeatures(args.embedding_size, args.dropout_chance)
    elif args.network == 'progressnet_boundingboxes':
        network = ProgressNetBoundingBoxes(args.embedding_size, args.dropout_chance, device)
    elif args.network == 'progressnet_categories':
        network = ProgressNetCategories(args.embedding_size, args.num_categories, args.dropout_chance)
    elif args.network == 'progressnet_boundingboxes_2d':
        network = ProgressNetBoundingBoxes2D(args.embedding_size, args.dropout_chance, device)
    elif args.network == 'progressnet_boundingboxes_vgg':
        network = ProgressNetBoundingBoxesVGG(args, device)
    elif args.network == 'progressnet_features_2d':
        network = ProgressNetFeatures2D(args.embedding_size, args.dropout_chance)
    elif args.network == 'progressnet_resnet':
        network = ProgressResNet()
    elif args.network == 'progressnet_resnet_indices':
        network = ProgressResNetIndices(device)
    elif args.network == 'dumb_static':
        network = StaticNet(device)
    elif args.network == 'dumb_random':
        network = RandomNet(device)

    if args.initialisation == 'xavier':
        network.apply(init_weights)
    return network.to(device)
