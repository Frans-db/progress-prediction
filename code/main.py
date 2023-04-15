import torch
import torch.optim as optim
import torch.nn as nn
import argparse
import wandb
import json
import os

from utils import parse_args, get_device, init
from datasets import get_datasets
from networks import get_network


def train():
    pass

def main() -> None:
    args = parse_args()
    device = get_device(args.device)
    init(args)

    # TODO: Data augmentations
    train_set, test_set, train_loader, test_loader = get_datasets(args)
    network = get_network(args, device)

    # get optimizer & scheduler & losses
    optimizer = optim.Adam(network.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_decay_every, gamma=args.lr_decay)
    l1_criterion = nn.L1Loss(reduction='none')
    l2_criterion = nn.MSELoss(reduction='none')

    # training
    iteration = 0
    done = False
    while not done:
        for batch in train_loader:
            pass

            if iteration % args.test_every == 0:
                for batch in test_loader:
                    pass

            iteration += 1
            scheduler.step()
            if iteration > args.iterations:
                done = True
                break

if __name__ == '__main__':
    main()
