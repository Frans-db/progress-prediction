import torch
import torch.nn.functional as F
from torchvision import models
from torch import nn

from .pyramidpooling import SpatialPyramidPooling

class ProgressNetFlat(nn.Module):
    def __init__(self, backbone_path: str = None) -> None:
        super().__init__()
        self.backbone = models.vgg16().features
        if backbone_path:
            self.backbone.load_state_dict(torch.load(backbone_path))


    def forward(self, frames: torch.FloatTensor, boxes: torch.FloatTensor) -> torch.FloatTensor:
        B, C, H, W = frames.shape
        frames = self.backbone(frames)

        return x

    def embed(self, x):
        x = self.fc1(x)
        x = F.sigmoid(x)
        x = self.fc2(x)
        return x

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


class ProgressNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()