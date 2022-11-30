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

from datasets import ProgressDataset
from datasets.transforms import ImglistToTensor, SwapDimensions
from networks import S3D

def get_device():
    return torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=5)

    parser.add_argument('--num_segments', type=int, default=1)
    parser.add_argument('--frames_per_segment', type=int, default=10)
    parser.add_argument('--sample_every_n_frames', type=int, default=1)
    
    return parser.parse_args()

def main():
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    args = parse_arguments()

    num_frames = args.num_segments * (args.frames_per_segment // args.sample_every_n_frames)

    dataset = ProgressDataset(
            './data/toy',
            num_segments=args.num_segments,
            frames_per_segment=args.frames_per_segment,
            every_nth_frame=args.sample_every_n_frames,
            transform=transforms.Compose([
                ImglistToTensor(),
                SwapDimensions()
            ]),
            )      

    dataloader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    net = S3D(num_classes=num_frames).to(get_device())
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=10**(-4))

    for epoch in range(args.epochs):
        epoch_loss = 0
        for i,(videos, labels) in enumerate(tqdm(dataloader)):
            num_samples = videos.shape[0]
            labels = torch.stack(labels).reshape(num_samples, num_frames).float()

            videos = videos.to(get_device())
            labels = labels.to(get_device())

            outputs = net(videos)

            optimizer.zero_grad()
            loss = criterion(outputs, labels)
            loss.backward()
            epoch_loss += loss.item()
            optimizer.step()
        print(f'[{epoch:2d}] loss: {epoch_loss:.3f}')

    net.eval()
    for i in range(20):
        video, labels = dataset[i]
        video = video.reshape(3, num_frames, 42, 42)
        predictions = net(video.unsqueeze(0)).squeeze()
        print(predictions)

        plt.plot(predictions.cpu().detach().numpy(), label='Predicted')
        plt.plot(labels, label='Actual')
        plt.title(f'Predicted vs Actul Completion Percentage - Video {i:5d}')
        plt.legend(loc='best')
        plt.savefig(f'./results/{i}.png')
        plt.clf()

if __name__ == '__main__':
    main()