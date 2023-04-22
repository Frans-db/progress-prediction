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


def bo(p, p_hat, device):
    m = torch.full(p.shape, 0.5).to(device)
    r = torch.full(p.shape, 0.5 * math.sqrt(2)).to(device)

    weight = ((p - m) / r).square() + ((p_hat - m) / r).square()
    return torch.clamp(weight, max=1)


def train(batch: Tuple, network: nn.Module, args: argparse.Namespace, device: torch.device, optimizer=None) -> dict:
    l1_criterion, l1_criterion_mean = nn.L1Loss(reduction='none'), nn.L1Loss(reduction='mean')
    l2_criterion, l2_criterion_mean = nn.MSELoss(reduction='none'), nn.MSELoss(reduction='mean')
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
    bo_weight = bo(predicted_progress, progress, device)
    count = progress.shape[-1]
    # optimizer
    if optimizer:
        optimizer.zero_grad()
        if args.loss == 'l1':
            loss =l1_criterion_mean(predicted_progress, progress)
        elif args.loss == 'l2':
            loss =l2_criterion_mean(predicted_progress, progress)
        loss.backward()
        optimizer.step()

    return {
        'video_names': video_names,
        'predictions': predicted_progress.cpu().detach(),
        'progress': progress.cpu().detach(),
        'l1_loss': l1_loss.sum().item(),
        'l1_bo_loss': (l1_loss * bo_weight).sum().item(),
        'l2_loss': l2_loss.sum().item(),
        'count': count
    }


def get_empty_result() -> dict:
    return {
        'l1_loss': 0.0,
        'l1_bo_loss': 0.0,
        'l2_loss': 0.0,
        'count': 0
    }


def update_result(result: dict, batch_result: dict) -> None:
    for key in result:
        if key in batch_result:
            result[key] += batch_result[key]


def wandb_log(result: dict, iteration: int, prefix: str) -> None:
    wandb.log({
        f'{prefix}_l1_loss': result['l1_loss'] / result['count'],
        f'{prefix}_l1_bo_loss': result['l1_bo_loss'] / result['count'],
        f'{prefix}_l2_loss': result['l2_loss'] / result['count'],
        'count': result['count'],
        'iteration': iteration,
    })


def main() -> None:
    args = parse_args()
    device = get_device(args.device)
    init(args)

    if args.experiment_name:
        experiment_root = os.path.join(args.data_root, 'experiments', args.experiment_name)

    train_set, test_set, train_loader, test_loader = get_datasets(args)
    network = get_network(args, device)
    model_parameters = filter(lambda p: p.requires_grad, network.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    # resnet:               11,255,201
    # vgg (frozen):         2,610,145
    # vgg:                  23,094,049
    # torchsummary.summary(network, (10, 3, 320, 240), boxes = None)
    # return
    # return
    # get optimizer & scheduler
    optimizer = optim.Adam(network.parameters(), lr=args.lr, betas=(args.beta1, args.beta2), weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_decay_every, gamma=args.lr_decay)

    # training
    iteration = 0
    done = False
    train_result, test_result = get_empty_result(), get_empty_result()
    # torch.backends.cudnn.benchmark = True
    while not done:
        for batch in train_loader:
            # train step
            batch_result = train(batch, network, args, device, optimizer=optimizer)
            if not args.wandb_disable:
                wandb_log(batch_result, iteration, 'train')
            update_result(train_result, batch_result)

            # test
            if iteration % args.test_every == 0:
                network.eval()
                with torch.no_grad():
                    test_json = []
                    for batch in test_loader:
                        # print(batch[1].shape)
                        batch_result = train(batch, network, args, device)
                        update_result(test_result, batch_result)
                        test_json.append({
                            'video_name': batch_result['video_names'][0],
                            'progress': batch_result['progress'][0].tolist(),
                            'predictions': batch_result['predictions'][0].tolist(),
                        })
                    if not args.wandb_disable:
                        wandb_log(test_result, iteration, 'test')
                        wandb_log(train_result, iteration, 'avg_train')
                    train_result, test_result = get_empty_result(), get_empty_result()
                    if args.experiment_name:
                        model_path = os.path.join(experiment_root, 'models', f'model_{iteration}.pth')
                        json_path = os.path.join(experiment_root, 'results', f'{iteration}.json')
                        with open(json_path, 'w+') as f:
                            json.dump(test_json, f)
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
