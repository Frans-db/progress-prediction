import torch
from torch import nn
from torchvision import models

class ToyNet(nn.Module):
    def __init__(self, dropout_chance: float) -> None:
        super().__init__()
        # self.backbone = models.resnet18()
        # self.backbone.fc = nn.Identity()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 32)

        self.backbone_dropout = nn.Dropout(p=dropout_chance)
        self.lstm = nn.LSTM(32, 32, batch_first=True)
        self.fc8 = nn.Linear(32, 1)

    def forward(self, frames: torch.FloatTensor) -> torch.FloatTensor:
        B, S, _, _, _ = frames.shape
        frames = torch.flatten(frames, end_dim=1)

        frames = torch.relu(self.pool(self.conv1(frames)))
        frames = torch.relu(self.pool(self.conv2(frames)))
        frames = torch.flatten(frames, start_dim=1)
        frames = torch.relu(self.fc1(frames))
        frames = torch.relu(self.fc2(frames))

        frames = frames.reshape(B, S, -1)
        frames, _ = self.lstm(frames)
        frames = torch.sigmoid(self.fc8(frames))
        
        return frames.reshape(B, S)
    
class ResNet(nn.Module):
    def __init__(self, backbone: str, backbone_path: str = None, dropout_chance: float = 0.5) -> None:
        super().__init__()
        if backbone == "resnet18":
            self.backbone = models.resnet18()
            self.backbone.fc = nn.Identity()
            self.fc = nn.Linear(512, 1)
        elif backbone == "resnet152":
            self.backbone = models.resnet152()
            self.backbone.fc = nn.Identity()
            self.fc = nn.Linear(2048, 1)
        else:
            raise Exception(f"Backbone {backbone} cannot be used for RSDNetFlat")

        if backbone_path:
            self.backbone.load_state_dict(torch.load(backbone_path))

        self.backbone_dropout = nn.Dropout(p=dropout_chance)
        self.fc7 = nn.Linear(512, 32)
        self.fc7_dropout = nn.Dropout(p=dropout_chance)
        self.lstm = nn.LSTM(32, 32, batch_first=True)
        self.fc8 = nn.Linear(32, 1)

    def forward(self, frames: torch.FloatTensor) -> torch.FloatTensor:
        B, S, _, _, _ = frames.shape
        frames = torch.flatten(frames, end_dim=1)

        frames = torch.relu(self.backbone_dropout(self.backbone(frames)))
        frames = torch.relu(self.fc7_dropout(self.fc7(frames)))
        frames = frames.reshape(B, S, -1)
        frames, _ = self.lstm(frames)
        frames = torch.sigmoid(self.fc8(frames))
        
        return frames.reshape(B, S)
