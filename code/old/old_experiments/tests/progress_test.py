"""
Toy data with progress dataset
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

class Net(nn.Module):
    def __init__(self, num_frames: int = 90, debug: bool = False) -> None:
        super().__init__()
        self.debug = debug
        self.conv1 = nn.Conv3d(3, 6, 5)
        self.pool = nn.MaxPool3d(2, 2)
        self.conv2 = nn.Conv3d(6, 16, 5)
        self.fc1 = nn.Linear(16*3*7*7, 120)
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


class ImglistToTensor():
    def __call__(self, img_list: List[Image.Image]):
        return torch.stack([transforms.functional.to_tensor(pic) for pic in img_list], dim=1)


class ProgressDataset(torch.utils.data.Dataset):
    def __init__(self,
                 root_path: str,
                 num_segments: int,
                 frames_per_segment: int,
                 imagefile_template: str = 'img_{:05d}.png',
                 test_mode: bool = False,
                 transform=None) -> None:
        super().__init__()
        self.root_path = root_path
        self.items = os.listdir(self.root_path)
        self.num_segments = num_segments
        self.frames_per_segment = frames_per_segment
        self.imagefile_template = imagefile_template
        self.transform = transform
        self.test_mode = test_mode

        self._check_samples()

    def _check_samples(self):
        """
        Check if each video has enough frames to be sampled
        """
        for video_name in self.items:
            path = os.path.join(self.root_path, video_name)
            frame_names = os.listdir(path)
            num_frames = len(frame_names)

            if num_frames <= 0:
                print(
                    f"\nDataset Warning: video {video_name} seems to have zero RGB frames on disk!\n")
            elif num_frames < self.num_segments * self.frames_per_segment:
                print(f"\nDataset Warning: video {video_name} has {num_frames} frames "
                      f"but the dataloader is set up to load "
                      f"(num_segments={self.num_segments})*(frames_per_segment={self.frames_per_segment})"
                      f"={self.num_segments * self.frames_per_segment} frames. Dataloader will throw an "
                      f"error when trying to load this video.\n")

    def _get_start_indices(self, num_frames: int) -> np.ndarray:
        """
        For each segment, choose a start index from where frames
        are to be loaded from.

        Args:
            record: VideoRecord denoting a video sample.
        Returns:
            List of indices of where the frames of each
            segment are to be loaded from.
        """
        # choose start indices that are perfectly evenly spread across the video frames.
        if self.test_mode:
            distance_between_indices = (
                num_frames - self.frames_per_segment + 1) / float(self.num_segments)

            start_indices = np.array([int(distance_between_indices / 2.0 + distance_between_indices * x)
                                      for x in range(self.num_segments)])
        # randomly sample start indices that are approximately evenly spread across the video frames.
        else:
            max_valid_start_index = (
                num_frames - self.frames_per_segment + 1) // self.num_segments
            start_indices = np.multiply(list(range(self.num_segments)), max_valid_start_index) + \
                np.random.randint(max_valid_start_index,
                                  size=self.num_segments)

        return start_indices

    def __getitem__(self, idx):
        item = self.items[idx]
        item_directory = os.path.join(self.root_path, item)
        image_files = sorted(os.listdir(item_directory))
        num_frames = len(image_files)

        start_indices = self._get_start_indices(num_frames)

        images = []
        labels = []
        for start_index in start_indices:
            indices = range(start_index, start_index+self.frames_per_segment)
            for i in indices:
                image_name = self.imagefile_template.format(i)
                image_path = os.path.join(item_directory, image_name)
                images.append(self._load_image(image_path))
                labels.append((i + 1) / num_frames)

        if self.transform is not None:
            return self.transform(images), np.array(labels)
        return images, np.array(labels)

    def _load_image(self, path: str) -> Image.Image:
        return Image.open(path).convert('RGB')

    def __len__(self):
        return len(self.items)


def imshow(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def get_device():
    return torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def main():
    num_frames = 24
    device = get_device()
    transform = transforms.Compose([
        ImglistToTensor()
    ])
    trainset = ProgressDataset('./data/toy', 1, num_frames, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                            shuffle=True, num_workers=2)

    net = Net(num_frames=num_frames).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(1):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(trainloader, 0):
            labels = labels.float()
            inputs = inputs.to(device) 
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 100 == 99:
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0

    with torch.no_grad():
        testset = ProgressDataset('./data/toy', 1, num_frames, transform=transform)
        num_videos = 10
        for video_index in range(num_videos):
            video, labels = testset[video_index]
            video = video.to(device)

            predictions = net(video.unsqueeze(0)).squeeze(0).cpu().numpy()

            plt.plot(labels, label='Real')
            plt.plot(predictions, label='Predictions')
            plt.savefig(f'./results/{video_index}.png')
            plt.clf()


if __name__ == '__main__':
    main()