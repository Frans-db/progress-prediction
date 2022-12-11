import torch.nn as nn
import torch.nn.functional as F
import torch

class Basic3D(nn.Module):
    def __init__(self, num_frames: int = 90, debug: bool = False) -> None:
        super().__init__()
        self.debug = debug
        self.conv1 = nn.Conv3d(3, 6, 5)
        self.pool = nn.MaxPool3d(2, 2)
        self.conv2 = nn.Conv3d(6, 16, 5)
        self.fc1 = nn.Linear(16*(((num_frames - 4) // 2 - 4) // 2)*7*7, 120)
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

class Basic2D(nn.Module):
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