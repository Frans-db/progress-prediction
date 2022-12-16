import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torch.optim as optim
import argparse
import uuid
import os
import random
import numpy as np

from datasets import ProgressDataset
from datasets.transforms import ImglistToTensor
from networks import S3D, Conv3D, LSTMNetwork
from utils import parse_arguments, get_device, set_seeds

def get_datasets(args):
    trainset = ProgressDataset(
        f'./data/{args.dataset}',
        num_videos=800,
        offset=0,
        num_segments=args.num_segments,
        frames_per_segment=args.frames_per_segment,
        sample_every=args.sample_every,
        transform=transforms.Compose([
            ImglistToTensor(),
        ])
    )
    testset = ProgressDataset(
        f'./data/{args.dataset}',
        num_videos=90,
        offset=800,
        num_segments=args.num_segments,
        frames_per_segment=args.frames_per_segment,
        sample_every=args.sample_every,
        transform=transforms.Compose([
            ImglistToTensor(),
        ]),
    )

    trainloader = DataLoader(
        dataset=trainset, 
        batch_size=args.batch_size,
        shuffle=True, 
        num_workers=args.num_workers
    )
    testloader = DataLoader(
        dataset=testset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )

    return trainset, trainloader, testset, testloader

def epoch(net, videos, labels, criterion, device, optimizer=None):
    videos = videos.to(device)
    labels = labels.float().to(device)

    outputs = net(videos).squeeze(-1)

    if optimizer is not None:
        optimizer.zero_grad()

    loss = criterion(outputs, labels)

    if optimizer is not None:
        loss.backward()
        optimizer.step()

    return loss.item()

def main():
    device = get_device()
    args = parse_arguments()
    num_frames = (args.num_segments * args.frames_per_segment) // args.sample_every

    set_seeds(args.seed)

    print(f'[Experiment {args.name}]')
    print(f'[Running on {device}]')
    print(f'[Seed {args.seed}]')
    print(f'[{num_frames} frames per sample]')

    trainset, trainloader, testset, testloader = get_datasets(args)

    if args.model == 'conv3d':
        net = Conv3D(
            intermediate_num_frames=args.intermediate_size,
            temporal_kernel_size=args.temporal_dimension
        ).to(device)
    elif args.model == 's3d':
        net = S3D(num_classes=num_frames).to(device)
    elif args.model == 'lstm':
        net = LSTMNetwork().to(device)

    criterion = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for epoch_index in range(args.epochs):
        train_loss = 0
        test_loss = 0

        for i, (videos, labels) in enumerate(trainloader):
            loss = epoch(net, videos, labels, criterion, device, optimizer=optimizer)
            train_loss += loss
        for i, (videos, labels) in enumerate(testloader):
            loss = epoch(net, videos, labels, criterion, device)
            test_loss += loss

        print(f'[{epoch_index:2d}] train loss: {train_loss:.4f} test loss: {test_loss:.4f}')

    """
    Problem:
    Currently no way to get all possible inputs from a video, so no way to predict the full progress graph
    Solutions:
    - Hack it into the current ProgressDataset using a counter
    - New dataset (Progressive Temporal Sampling?)
    - 
    """
    if not os.path.isdir(f'./results/{args.name}'):
        os.mkdir(f'./results/{args.name}')
    net.eval()
    testset.mode = 'visualise'
    with torch.no_grad():
        for i in range(20):
            labels = []
            predictions = []
            testset.counter = 0
            counter = 0
            next_counter = 1
            while next_counter >= counter:
                data = testset[i]
                video = data[0].to(device)
                label = data[1]

                prediction = net(video.unsqueeze(0)).squeeze().item()
                predictions.append(prediction)
                labels.append(label)

                counter = next_counter
                next_counter = testset.counter


            plt.plot(predictions, label='Predicted')
            plt.plot(labels, label='Actual')
            plt.title(f'Predicted vs Actul Completion Percentage - Video {i:5d}')
            plt.legend(loc='best')
            plt.savefig(f'./results/{args.name}/{i}.png')
            plt.clf()


if __name__ == '__main__':
    main()
