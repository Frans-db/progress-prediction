import torch
import torch.nn.functional as F
from torchvision import models
from torch import nn

from .pyramidpooling import SpatialPyramidPooling


class RSDNetFlat(nn.Module):
    def __init__(self, backbone: str, backbone_path: str = None) -> None:
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

    def forward(self, frames: torch.FloatTensor) -> torch.FloatTensor:
        frames = self.backbone(frames)
        return torch.sigmoid(self.fc(frames))
    
    def embed(self, frames: torch.FloatTensor) -> torch.FloatTensor:
        return self.backbone(frames)

class LSTMNet(nn.Module):
    def __init__(self, feature_dim: int, dropout_chance: float, finetune: bool) -> None:
        super().__init__()

        self.cnn_dropout = nn.Dropout(p=dropout_chance)
        self.lstm_dropout = nn.Dropout(p=dropout_chance)
        
        self.lstm1 = nn.LSTM(feature_dim, 512, batch_first=True)

        if finetune:
            for param in self.parameters():
                param.requires_grad = False
    
        self.fc_progress = nn.Linear(512, 1)

    def forward(self, data: torch.FloatTensor) -> torch.FloatTensor:
        B, S, _ = data.shape
        data = self.cnn_dropout(data)
        data, _ = self.lstm1(data)

        data = self.lstm_dropout(data)

        progress = torch.sigmoid(self.fc_progress(data))

        return progress.reshape(B, S)    

class RSDNet(nn.Module):
    def __init__(self, feature_dim: int, rsd_normalizer: float, dropout_chance: float, finetune: bool) -> None:
        super().__init__()
        self.rsd_normalizer = rsd_normalizer

        self.cnn_dropout = nn.Dropout(p=dropout_chance)
        self.lstm_dropout = nn.Dropout(p=dropout_chance)

        self.lstm1 = nn.LSTM(feature_dim, 512, batch_first=True)

        if finetune:
            for param in self.parameters():
                param.requires_grad = False

        self.fc_rsd = nn.Linear(512 + 1, 1)
        self.fc_progress = nn.Linear(512 + 1, 1)

    def forward(self, frames: torch.FloatTensor, elapsed: torch.FloatTensor) -> torch.FloatTensor:
        B, S = elapsed.shape
        frames = self.cnn_dropout(frames)
        frames, _ = self.lstm1(frames)
        frames = self.lstm_dropout(frames)

        elapsed = elapsed.reshape(B, S, 1)
        frames = torch.cat((frames, elapsed), dim=-1)
        frames = frames.reshape(B*S, -1)

        rsd = self.fc_rsd(frames)
        progress = torch.sigmoid(self.fc_progress(frames))

        return rsd.reshape(B, S), progress.reshape(B, S)