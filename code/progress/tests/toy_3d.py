"""
Treating toy dataset as 3D videos, and regressing to a list of completion percentages
"""

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import random
from typing import List


class ImglistToTensor():
    def __call__(self, img_list: List[Image.Image]):
        return torch.stack([transforms.functional.to_tensor(pic) for pic in img_list], dim=1)


class Toy3DDataset(Dataset):
    def __init__(self, root_dir: str, num_videos: int, frames_per_video: int = 90, transform=None, shuffle: bool = False) -> None:
        super().__init__()
        self.root_dir = root_dir
        self.num_videos = num_videos
        self.frames_per_video = frames_per_video
        self.transform = transform

        self.indices = list(range(0, len(self)))
        if shuffle:
            random.shuffle(self.indices)

    def __len__(self):
        return self.num_videos

    def __getitem__(self, index):
        new_index = self.indices[index]
        images = []
        labels = []
        for frame_index in range(self.frames_per_video):
            path = f'{self.root_dir}/{new_index:05d}/img_{frame_index:05d}.png'
            images.append(Image.open(path).convert('RGB'))
            labels.append((frame_index + 1) / self.frames_per_video)

        if self.transform:
            images = self.transform(images)

        return images, np.array(labels)

def imshow(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def get_device():
    return torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def main():
    class Net(nn.Module):
        def __init__(self, num_frames: int = 90, debug: bool = False) -> None:
            super().__init__()
            self.debug = debug
            self.conv1 = nn.Conv3d(3, 6, 5)
            self.pool = nn.MaxPool3d(2, 2)
            self.conv2 = nn.Conv3d(6, 16, 5)
            self.fc1 = nn.Linear(16*19*7*7, 120)
            self.fc2 = nn.Linear(120, 100)
            self.fc3 = nn.Linear(100, num_frames)

        def forward(self, x):
            if self.debug: print(x.shape)
            x = self.pool(F.relu(self.conv1(x)))
            if self.debug: print(x.shape)
            x = self.pool(F.relu(self.conv2(x)))
            if self.debug: print(x.shape)
            x = torch.flatten(x, 1)
            if self.debug: print(x.shape)
            x = F.relu(self.fc1(x))
            if self.debug: print(x.shape)
            x = F.relu(self.fc2(x))
            if self.debug: print(x.shape)
            x = self.fc3(x)
            if self.debug: print(x.shape)
            return x

    device = get_device()
    transform = transforms.Compose([
        ImglistToTensor()
    ])
    trainset = Toy3DDataset('./data/toy', num_videos=891, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                            shuffle=True, num_workers=2)

    net = Net(debug=False).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(3):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(trainloader, 0):
            # batch_size = labels.shape[0]
            labels = labels.float()
            inputs = inputs.to(device) 
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 500 == 499:
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0

    with torch.no_grad():
        testset = Toy3DDataset('./data/toy', num_videos=891, transform=transform)
        num_videos = 10
        for video_index in range(num_videos):
            video, labels = testset[video_index]
            video = video.to(device)

            predictions = net(predictions.unsqueeze(0)).squeeze(0).cpu().item()

            plt.plot(labels, label='Real')
            plt.plot(predictions, label='Predictions')
            plt.savefig(f'./results/{video_index}.png')
            plt.clf()
            plt.show()


if __name__ == '__main__':
    main()