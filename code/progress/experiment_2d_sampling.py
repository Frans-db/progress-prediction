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

from datasets import VideoDataset, VideoFrameDataset, ProgressDataset
from networks import Conv2D
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
        transform=ImglistToTensor(dim=0)
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
        transform=ImglistToTensor(dim=0)
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

    trainset, trainloader, testset, testloader, videoset = get_datasets(args)

    print(f'[Experiment {args.name}]')
    print(f'[Running on {device}]')
    print(f'[Seed {args.seed}]')
    print(f'[{len(trainset)} train frames]')

    net = Conv2D().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001,  momentum=0.9)


    for epoch in range(args.epochs):
        train_loss = 0
        test_loss = 0
        for i, (inputs, labels) in enumerate(trainloader):
            batch_size, T, C, W, H = inputs.shape

            labels = labels.reshape(batch_size * T, 1).float()
            inputs = inputs.reshape(batch_size * T, C, W, H)

            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
        for i, (inputs, labels) in enumerate(testloader):
            batch_size, T, C, W, H = inputs.shape

            labels = labels.reshape(batch_size * T, 1).float()
            inputs = inputs.reshape(batch_size * T, C, W, H)

            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = net(inputs)
            loss = criterion(outputs, labels)

            test_loss += loss.item()

        print(f'[{epoch:2d}] train loss: {train_loss:.4f} test loss: {test_loss:.4f}')

    if not os.path.isdir(f'./results/experiments/2d_sampled/{args.name}'):
        os.mkdir(f'./results/experiments/2d_sampled/{args.name}')

    net.eval()
    with torch.no_grad():
        for video_index in range(20):
            video, labels = videoset[video_index]

            num_frames = video.shape[1]

            predictions = []
            for i in range(num_frames):
                frame = video[:, i, :, :]

                frame = frame.to(device)
                prediction = net(frame.unsqueeze(0)).squeeze(0).cpu().item()
                predictions.append(prediction)

            plt.plot(labels, label='Real')
            plt.plot(predictions, label='Predictions')
            plt.xlabel('Frame Number')
            plt.ylabel('Progression (%)')
            plt.title('Real vs Predicted progression')
            plt.legend(loc='best')
            plt.savefig(f'./results/experiments/2d_sampled/{args.name}/{video_index}.png')
            plt.clf()


if __name__ == '__main__':
    main()
