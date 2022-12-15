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




def main():
    device = get_device()
    args = parse_arguments()
    num_frames = (args.num_segments * args.frames_per_segment) // args.sample_every

    set_seeds(args.seed)

    print(f'[Experiment {args.name}]')
    print(f'[Running on {device}]')
    print(f'[Seed {args.seed}]')
    print(f'[{num_frames} frames per sample]')

    trainset = ProgressDataset(
        f'./data/{args.dataset}',
        num_segments=args.num_segments,
        frames_per_segment=args.frames_per_segment,
        sample_every=args.sample_every,
        transform=transforms.Compose([
            ImglistToTensor(),
        ])
    )
    testset = ProgressDataset(
        f'./data/{args.dataset}',
        num_segments=args.num_segments,
        frames_per_segment=args.frames_per_segment,
        sample_every=args.sample_every,
        transform=transforms.Compose([
            ImglistToTensor(),
        ]),
        test_mode=False,
    )

    trainloader = DataLoader(
        dataset=trainset, 
        batch_size=args.batch_size,
        shuffle=True, 
        num_workers=args.num_workers
    )

    if args.model == 'conv3d':
        net = Conv3D(num_frames=num_frames).to(device)
    elif args.model == 's3d':
        net = S3D(num_classes=num_frames).to(device)
    elif args.model == 'lstm':
        net = LSTMNetwork().to(device)

    criterion = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(args.epochs):
        epoch_loss = 0
        for i, (videos, labels) in enumerate(trainloader, 0):
            videos = videos.to(device)
            labels = labels.float().to(device)


            outputs = net(videos)

            optimizer.zero_grad()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
        print(f'[{epoch:2d}] loss: {epoch_loss:.4f}')

    if not os.path.isdir(f'./results/{args.name}'):
        os.mkdir(f'./results/{args.name}')
    net.eval()
    with torch.no_grad():
        for i in range(20):
            video = testset[i][0].to(device)
            labels = testset[i][1]
            predictions = net(video.unsqueeze(0)).squeeze()

            plt.plot(predictions.cpu().detach().numpy(), label='Predicted')
            plt.plot(labels, label='Actual')
            plt.title(f'Predicted vs Actul Completion Percentage - Video {i:5d}')
            plt.legend(loc='best')
            plt.savefig(f'./results/{args.name}/{i}.png')
            plt.clf()


if __name__ == '__main__':
    main()
