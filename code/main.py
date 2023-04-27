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

from utils import parse_args, get_device, init
from datasets import get_datasets
from networks import get_network

def train(batch: Tuple, network: nn.Module, args: argparse.Namespace, device: torch.device, optimizer=None) -> dict:
    l1_criterion = nn.L1Loss(reduction='none')
    l2_criterion = nn.MSELoss(reduction='none')
    # extract data from batch
    num_items = len(batch)
    video_names = batch[0]
    data = batch[1:num_items-2]
    data = tuple(map(lambda x: x.to(device), data))
    # forward pass
    predicted_progress = network(*data)
    # loss calculations
    progress = batch[-2].to(device)
    mask = (progress != 0).int()
    l1_loss = l1_criterion(predicted_progress, progress) * mask
    l2_loss = l2_criterion(predicted_progress, progress) * mask
    count = batch[-1].sum()
    # optimizer
    if optimizer:
        optimizer.zero_grad()
        (l2_loss.sum() / count).backward()
        optimizer.step()

    return {
        'video_names': video_names,
        'predictions': predicted_progress.cpu().detach(),
        'progress': progress.cpu().detach(),
        'l1_loss': l1_loss.sum().item(),
        'l2_loss': l2_loss.sum().item(),
        'count': count.item()
    }


def train_rsd_images(batch: Tuple, network: nn.Module, args: argparse.Namespace, device: torch.device, optimizer=None) -> dict:
    l1_criterion = nn.L1Loss(reduction='none')
    l2_criterion = nn.MSELoss(reduction='none')
    # extract data from batch
    num_items = len(batch)
    video_names = batch[0]
    data = batch[1:num_items-1]
    data = tuple(map(lambda x: x.to(device), data))
    # forward pass
    predicted_progress = network(*data)
    # loss calculations
    progress = batch[-1].to(device)
    l1_loss = l1_criterion(predicted_progress, progress)
    l2_loss = l2_criterion(predicted_progress, progress)
    count = batch[-1].shape[0]
    # optimizer
    if optimizer:
        optimizer.zero_grad()
        (l2_loss.sum() / count).backward()
        optimizer.step()

    return {
        'video_names': video_names,
        'predictions': predicted_progress.cpu().detach(),
        'progress': progress.cpu().detach(),
        'l1_loss': l1_loss.sum().item(),
        'l2_loss': l2_loss.sum().item(),
        'count': count
    }

def get_empty_result() -> dict:
    return {
        'l1_loss': 0.0,
        'l2_loss': 0.0,
        'count': 0
    }


def update_result(result: dict, batch_result: dict) -> None:
    for key in result:
        if key in batch_result:
            result[key] += batch_result[key]


def wandb_log(result: dict, iteration: int, prefix: str, commit: bool = True) -> None:
    wandb.log({
        f'{prefix}_l1_loss': result['l1_loss'] / result['count'],
        f'{prefix}_l2_loss': result['l2_loss'] / result['count'],
        f'{prefix}_count': result['count'],
        'iteration': iteration,
    }, commit = commit)


def main() -> None:
    args = parse_args()
    device = get_device(args.device)
    init(args)

    if args.experiment_name:
        experiment_root = os.path.join(args.data_root, 'experiments', args.experiment_name)

    train_set, test_set, train_loader, test_loader = get_datasets(args)
    network = get_network(args, device)
    print('--- Network 🤖 ---')
    print(network)
    print('--- Datasets 💿 ---')
    print(f'Train size: {len(train_set)} ({len(train_loader)})')
    print(f'Test size: {len(test_set)} ({len(test_loader)})')

    if args.optimizer == 'adam':
        optimizer = optim.Adam(network.parameters(), lr=args.lr, betas=(args.beta1, args.beta2), weight_decay=args.weight_decay)
    elif args.optimizer == 'sgd':
        optimizer = optim.SGD(network.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_decay_every, gamma=args.lr_decay)
    if args.wandb_watch and not args.wandb_disable and not args.debug:
        print('-> Watching model 👀')
        wandb.watch(network)

    # training
    iteration = 0
    done = False
    train_result, test_result = get_empty_result(), get_empty_result()
    while not done:
        for batch in train_loader:
            # train step
            # TODO: Try to combine both train methods (or seperate files for progressnet/rsdnet)
            if args.dataset_type == 'images':
                batch_result = train_rsd_images(batch, network, args, device, optimizer=optimizer)
            else:
                batch_result = train(batch, network, args, device, optimizer=optimizer)
            update_result(train_result, batch_result)
            # log average train results
            if iteration % args.log_every == 0 and iteration > 0:
                if not args.wandb_disable and not args.debug:
                    commit = iteration % args.test_every != 0
                    wandb_log(train_result, iteration, 'train', commit=commit)
                train_result = get_empty_result()
            # test
            if iteration % args.test_every == 0 and iteration > 0:
                network.eval()
                with torch.no_grad():
                    test_json = []
                    for batch in test_loader:
                        # TODO: Try to combine both train methods
                        if args.dataset_type == 'images':
                            batch_result = train_rsd_images(batch, network, args, device)
                        else:
                            batch_result = train(batch, network, args, device)
                        update_result(test_result, batch_result)
                        # test_json.append({
                        #     'video_name': batch_result['video_names'][0],
                        #     'progress': batch_result['progress'][0].tolist(),
                        #     'predictions': batch_result['predictions'][0].tolist(),
                        # })
                    if not args.wandb_disable and not args.debug:
                        wandb_log(test_result, iteration, 'test')
                    test_result = get_empty_result()
                    if args.experiment_name:
                        model_path = os.path.join(experiment_root, 'models', f'model_{iteration}.pth')
                        json_path = os.path.join(experiment_root, 'results', f'{iteration}.json')
                        # with open(json_path, 'w+') as f:
                        #     json.dump(test_json, f)
                        torch.save(network.state_dict(), model_path)
                network.train()
            # update iteration & scheduler
            iteration += 1
            scheduler.step()
            if iteration > args.iterations:
                done = True
                break


if __name__ == '__main__':
    main()
