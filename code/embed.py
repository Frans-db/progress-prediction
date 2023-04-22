import torch
import torch.optim as optim
import torch.nn as nn
import argparse
import wandb
import json
import os
from typing import Tuple
import math
import numpy as np
from tqdm import tqdm

from utils import parse_args, get_device, init
from datasets import get_datasets
from networks import get_network

def add_arg(parser):
    parser.add_argument('--save_directory', type=str, required=True)
    parser.add_argument('--split_size', type=int, default=100)

    return parser.parse_args()

def main() -> None:
    parser = parse_args(parse=False)
    args = add_arg(parser)
    args.wandb_disable = True
    args.experiment_name = None
    args.bounding_boxes = True
    args.subsection_chance = 0.0
    args.subsample_chance = 0.0
    args.dropout_chance = 0.0

    device = get_device(args.device)
    init(args)

    train_set, test_set, train_loader, test_loader = get_datasets(args)
    network = get_network(args, device).vgg
    print(network)
    print(args.basemodel, args.basemodel_name)

    save_root = os.path.join(args.data_root, args.dataset, args.save_directory)
    os.mkdir(save_root)

    network.eval()
    with torch.no_grad():
        embed(network, train_loader, device, save_root)
        embed(network, test_loader, device, save_root)

def make_dir(name: str, root: str):
    directory = name.split('/')[0]
    path = os.path.join(root, directory)
    if not os.path.isdir(path):
        os.mkdir(path)

def embed(network, loader, device, root: str):
    for i, (video_names, frames, bounding_boxes, progress) in enumerate(tqdm(loader)):
        name = f'{video_names[0]}.txt'
        make_dir(name, root)
        B, S, C, H, W = frames.shape
        frames = frames.reshape(B*S, C, H, W)
        split = torch.split(frames, 100, dim=0)

        embeddings = []
        for t in split:
            t = t.to(device)
            num_samples = t.shape[0]
            embedded = network(t).reshape(num_samples, -1).cpu()
            embeddings.append(embedded)

        embedded = torch.cat(embeddings, dim=0)
        embedded = embedded.tolist()
    
        rows = []
        for row in embedded:
            rows.append(' '.join(list(map(str, row))))
        txt = '\n'.join(rows)
        with open(os.path.join(root, name), 'w+') as f:
            f.write(txt)

        



if __name__ == '__main__':
    main()
