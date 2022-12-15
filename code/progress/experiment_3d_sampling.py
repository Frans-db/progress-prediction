"""
Treating toy dataset as 2D images, and regressing to the completion percentage.
The idea is that even if this does not give perfect results, it should be able to approximate using
the number and its colour
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import uuid
import random

from datasets import VideoDataset, ProgressDataset
from networks import Conv3D
from datasets.transforms import ImglistToTensor
from utils import parse_arguments, get_device, set_seeds


def get_datasets(args):
    trainset = ProgressDataset(
        f'./data/{args.dataset}', 
        num_videos=800,
        offset=0,
        num_segments=args.num_segments, 
        frames_per_segment=args.frames_per_segment,
        sample_every=args.sample_every,
        transform=ImglistToTensor()
    )

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                              shuffle=True, num_workers=args.num_workers)

    testset = ProgressDataset(
        f'./data/{args.dataset}', 
        num_videos=90,
        offset=800,
        num_segments=args.num_segments, 
        frames_per_segment=args.frames_per_segment,
        sample_every=args.sample_every,
        transform=ImglistToTensor()
    )

    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
                                              shuffle=True, num_workers=args.num_workers)

    videoset = VideoDataset(
        f'./data/{args.dataset}', num_videos=90, offset=800, transform=ImglistToTensor())

    return trainset, trainloader, testset, testloader, videoset


def main():
    device = get_device()
    args = parse_arguments()
    set_seeds(args.seed)
    num_frames = args.num_segments * args.frames_per_segment

    trainset, trainloader, testset, testloader, videoset = get_datasets(args)

    print(f'[Experiment {args.name}]')
    print(f'[Running on {device}]')
    print(f'[Seed {args.seed}]')
    print(f'[{len(trainset)} train frames]')

    net = Conv3D(output_frames=num_frames, intermediate_num_frames=args.intermediate_size, temporal_kernel_size=args.temporal_dimension).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001,  momentum=0.9)


    for epoch in range(args.epochs):
        train_loss = 0
        test_loss = 0
        for i, (inputs, labels) in enumerate(trainloader):
            inputs = inputs.to(device)
            labels = labels.float().to(device)

            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        for i, (inputs, labels) in enumerate(testloader):
            inputs = inputs.to(device)
            labels = labels.float().to(device)

            outputs = net(inputs)
            loss = criterion(outputs, labels)

            test_loss += loss.item()

        print(f'[{epoch:2d}] train loss: {train_loss:.4f} test loss: {test_loss:.4f}')

    if not os.path.isdir(f'./results/experiments/3d_sampled/{args.name}'):
        os.mkdir(f'./results/experiments/3d_sampled/{args.name}')

    net.eval()
    with torch.no_grad():
        testset.test_mode = False
        for video_index in range(20):
            video, labels = testset[video_index]

            video = video.to(device)
            labels = labels.float().to(device)

            predictions = net(video.unsqueeze(0)).squeeze(0).cpu().numpy()

            plt.plot(labels, label='Real')
            plt.plot(predictions, label='Predictions')
            plt.xlabel('Frame Number')
            plt.ylabel('Progression (%)')
            plt.title('Real vs Predicted progression')
            plt.legend(loc='best')
            plt.savefig(f'./results/experiments/3d_sampled/{args.name}/{video_index}.png')
            plt.clf()

        testset.test_mode = True
        for video_index in range(20):
            video, labels = testset[video_index]

            video = video.to(device)
            labels = labels.float().to(device)

            predictions = net(video.unsqueeze(0)).squeeze(0).cpu().numpy()

            plt.plot(labels, label='Real')
            plt.plot(predictions, label='Predictions')
            plt.xlabel('Frame Number')
            plt.ylabel('Progression (%)')
            plt.title('Real vs Predicted progression')
            plt.legend(loc='best')
            plt.savefig(f'./results/experiments/3d_sampled/{args.name}/{video_index}_test.png')
            plt.clf()


if __name__ == '__main__':
    main()
