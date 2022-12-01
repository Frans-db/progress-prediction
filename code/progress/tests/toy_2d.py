"""
Treating toy dataset as 2D images, and regressing to the completion percentage.
The idea is that even if this does not give perfect results, it should be able to approximate using
the number and its colour
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

class Toy2DDataset(Dataset):
    def __init__(self, root_dir: str, num_videos: int, frames_per_video: int = 90, transform = None, shuffle: bool = False) -> None:
        super().__init__()
        self.root_dir = root_dir
        self.num_videos = num_videos
        self.frames_per_video = frames_per_video
        self.transform = transform

        self.indices = list(range(0, len(self)))
        if shuffle:
            random.shuffle(self.indices)


    def __len__(self):
        return self.num_videos * self.frames_per_video

    def __getitem__(self, index):
        new_index = self.indices[index]
        video_index = new_index // self.frames_per_video
        frame_index = new_index % self.frames_per_video

        path = f'{self.root_dir}/{video_index:05d}/img_{frame_index:05d}.png'
        image = Image.open(path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, (frame_index + 1) / self.frames_per_video

def imshow(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def get_device():
    return torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def main():
    device = get_device()
    print(f'[Running on {device}]')
    class Net(nn.Module):
        def __init__(self, debug: bool = False) -> None:
            super().__init__()
            self.debug = debug
            self.conv1 = nn.Conv2d(3, 6, 7)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(6, 16, 7)
            self.fc1 = nn.Linear(16 * 6 * 6, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 1)

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

    batch_size = 4

    transform = transforms.Compose(
        [transforms.ToTensor()])
    

    trainset = Toy2DDataset('./data/toy', num_videos=891, shuffle=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=2)

    # testset = torchvision.datasets.MNIST(root='./data', train=False,
    #                                     download=True, transform=transform)
    # testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
    #                                         shuffle=False, num_workers=2)

    net = Net().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001,  momentum=0.9)
    

    for epoch in range(2):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 2000 == 1999:
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0

    with torch.no_grad():
        testset = Toy2DDataset('./data/toy', num_videos=891, transform=transform)
        num_videos = 10
        for video_index in range(num_videos):
            predictions = []
            reals = []
            for i in range(90):
                video, label = testset[video_index*90 + i]
                video = video.to(device)
                
                reals.append(label)
                prediction = net(video.unsqueeze(0)).squeeze(0).cpu().item()
                predictions.append(prediction)
            plt.plot(reals, label='Real')
            plt.plot(predictions, label='Predictions')
            plt.show()
    # since we're not training, we don't need to calculate the gradients for our outputs
    # with torch.no_grad():
    #     for data in testloader:
    #         images, labels = data[0].to(device), data[1].to(device)
    #         # calculate outputs by running images through the network
    #         outputs = net(images)
    #         # the class with the highest energy is what we choose as prediction
    #         _, predicted = torch.max(outputs.data, 1)
    #         total += labels.size(0)
    #         correct += (predicted == labels).sum().item()

    # print(f'Accuracy of the network on the 10000 test images: {100 * correct     // total} %')

if __name__ == '__main__':
    main()