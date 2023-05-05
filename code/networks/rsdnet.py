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
            self.backbone.fc = nn.Linear(512, 1)
        elif backbone == "resnet152":
            self.backbone = models.resnet152()
            self.backbone.fc = nn.Linear(2048, 1)
        else:
            raise Exception(f"Backbone {backbone} cannot be used for RSDNetFlat")

        if backbone_path:
            self.backbone.load_state_dict(torch.load(backbone_path))

    def forward(self, frames: torch.FloatTensor) -> torch.FloatTensor:
        return torch.sigmoid(self.backbone(frames))


class RSDNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
