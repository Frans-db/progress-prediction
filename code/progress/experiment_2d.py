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
from datasets import Toy3DDataset, Toy2DDataset
from networks import Basic2D
from datasets.transforms import ImglistToTensor
import random


def imshow(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def get_device():
    return torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default=None)
    parser.add_argument('--seed', type=int, default=None)

    parser.add_argument('--dataset', type=str, default='toy')
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=1)

    args = parser.parse_args()
    if args.name is None:
        args.name = uuid.uuid4()
    if args.seed is None:
        args.seed = random.randint(0, 1_000_000_000)
    return args


def main():
    device = get_device()
    args = parse_arguments()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    print(f'[Running on {device}]')
    print(f'[Seed {args.seed}]')

    trainset = Toy2DDataset(
        f'./data/{args.dataset}', num_videos=800, transform=transforms.ToTensor())
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                              shuffle=False, num_workers=args.num_workers)

    net = Basic2D().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001,  momentum=0.9)

    for epoch in range(args.epochs):
        running_loss = 0.0
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

            running_loss += loss.item()
            if i % 2000 == 1999:
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0


    if not os.path.isdir(f'./results/{args.name}'):
        os.mkdir(f'./results/{args.name}')
    with torch.no_grad():
        testset = Toy3DDataset(f'./data/{args.dataset}', num_videos=90, offset=800, transform=ImglistToTensor())
        num_videos = 10
        for video_index in range(num_videos):
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
            plt.savefig(f'./results/{args.name}/{video_index}.png')
            plt.clf()


if __name__ == '__main__':
    main()
