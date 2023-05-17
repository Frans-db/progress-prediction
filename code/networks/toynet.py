import torch
from torch import nn
from torchvision import models


class ToyNet(nn.Module):
    def __init__(self, dropout_chance: float) -> None:
        super().__init__()
        # self.backbone = models.resnet18()
        # self.backbone.fc = nn.Identity()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 16, 3),
            nn.Conv2d(16, 16, 3),
            nn.MaxPool2d(3),
            nn.Conv2d(16, 32, 3),
            nn.Conv2d(32, 32, 3),
            nn.MaxPool2d(3),
            nn.Conv2d(32, 32, 2),
        )
        self.backbone_dropout = nn.Dropout(p=dropout_chance)
        self.fc7 = nn.Linear(32, 8)
        self.fc7_dropout = nn.Dropout(p=dropout_chance)
        self.lstm = nn.LSTM(8, 8, batch_first=True)
        self.fc8 = nn.Linear(8, 1)

    def forward(self, frames: torch.FloatTensor) -> torch.FloatTensor:
        B, S, _, _, _ = frames.shape
        frames = torch.flatten(frames, end_dim=1)
        frames = torch.relu(self.backbone_dropout(self.backbone(frames)))
        frames = frames.reshape(B, S, -1)

        frames = torch.relu(self.fc7_dropout(self.fc7(frames)))
        frames, _ = self.lstm(frames)
        frames = torch.sigmoid(self.fc8(frames))
        
        return frames.reshape(B, S)