import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torch.optim as optim
import argparse
import random
from tqdm import tqdm
import numpy as np

from datasets import ProgressDataset, Toy3DDataset
from datasets.transforms import ImglistToTensor
from networks import S3D, Basic3D


def get_device():
    return torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='toy')
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=5)

    parser.add_argument('--num_segments', type=int, default=1)
    parser.add_argument('--frames_per_segment', type=int, default=10)

    return parser.parse_args()


def main():
    device = get_device()
    args = parse_arguments()
    num_frames = 90 #args.num_segments * args.frames_per_segment

    # trainset = ProgressDataset(
    #     f'./data/{args.dataset}',
    #     num_segments=args.num_segments,
    #     frames_per_segment=args.frames_per_segment,
    #     transform=transforms.Compose([
    #         ImglistToTensor(),
    #     ])
    # )
    trainset = Toy3DDataset(
        root_dir=f'./data/{args.dataset}',
        num_videos=800,
        transform=transforms.Compose([
            ImglistToTensor(),
        ])
    )
    testset = Toy3DDataset(
        root_dir=f'./data/{args.dataset}',
        num_videos=92,
        offset=800,
        transform=transforms.Compose([
            ImglistToTensor(),
        ])
    )
    trainloader = DataLoader(
        dataset=trainset, 
        batch_size=args.batch_size,
        shuffle=True, 
        num_workers=args.num_workers
    )

    net = Basic3D(num_frames=num_frames).to(get_device())
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
        print(f'[{epoch:2d}] loss: {epoch_loss:.3f}')

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
            plt.savefig(f'./results/{i}.png')
            plt.clf()


if __name__ == '__main__':
    main()
