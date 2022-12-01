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