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

from datasets import VideoDataset, VideoFrameDataset
from networks import Conv2D
from datasets.transforms import ImglistToTensor
from utils import parse_arguments, get_device, set_seeds


def get_datasets(args):
    trainset = VideoFrameDataset(
        f'./data/{args.dataset}', num_videos=800, transform=transforms.ToTensor())

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                              shuffle=True, num_workers=args.num_workers)
    testset = VideoDataset(
        f'./data/{args.dataset}', num_videos=90, offset=800, transform=ImglistToTensor())

    return trainset, trainloader, testset

def main():
    device = get_device()
    args = parse_arguments()
    set_seeds(args.seed)

    trainset, trainloader, testset = get_datasets(args)

    print(f'[Experiment {args.name}]')
    print(f'[Running on {device}]')
    print(f'[Seed {args.seed}]')
    print(f'[{len(trainset)} frames]')

    net = Conv2D().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001,  momentum=0.9)

    for epoch in range(args.epochs):
        epoch_loss = 0
        for i, (inputs, labels) in enumerate(trainloader, 0):
            batch_size = labels.shape[0]
            labels = labels.reshape(batch_size, 1).float()
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
        print(f'[{epoch:2d}] loss: {epoch_loss:.4f}')

    if not os.path.isdir(f'./results/experiments/{args.name}'):
        os.mkdir(f'./results/experiments/{args.name}')
    net.eval()
    with torch.no_grad():
        for video_index in range(20):
            video, labels = testset[video_index]

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
            plt.savefig(f'./results/experiments/{args.name}/{video_index}.png')
            plt.clf()


if __name__ == '__main__':
    main()
