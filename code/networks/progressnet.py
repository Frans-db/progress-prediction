import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models
from torchvision.ops import roi_align
from typing import List

from .pyramidpooling import SpatialPyramidPooling


class ProgressNetFlat(nn.Module):
    def __init__(
        self,
        pooling_layers: List[int],
        roi_size: int,
        dropout_chance: float,
        embed_dim: int,
        backbone: str,
        backbone_path: str = None,
    ) -> None:
        super().__init__()
        if backbone == "vgg16":
            self.backbone = models.vgg16().features
        else:
            raise Exception(f"Backbone {backbone} cannot be used for ProgressnetFlat")

        if backbone_path:
            self.backbone.load_state_dict(torch.load(backbone_path))

        pooling_size = sum(map(lambda x: x**2, pooling_layers))

        self.spp = SpatialPyramidPooling(pooling_layers)
        self.spp_fc = nn.Linear(512 * pooling_size, embed_dim)
        self.spp_dropout = nn.Dropout(p=dropout_chance)

        self.roi_fc = nn.Linear(512 * roi_size**2, embed_dim)
        self.roi_dropout = nn.Dropout(p=dropout_chance)

        self.fc7 = nn.Linear(embed_dim * 2, 1)

    def forward(self, frames: torch.FloatTensor, boxes: torch.Tensor) -> torch.FloatTensor:
        B, C, H, W = frames.shape

        frames = self.vgg(frames)
        frames = self.vgg_dropout(frames)

        spp_pooled = self.spp(frames)
        spp_pooled = self.spp_fc(spp_pooled)
        spp_pooled = self.spp_dropout(spp_pooled)
        spp_pooled = torch.relu(spp_pooled)

        indices = torch.arange(0, B, device=self.device).reshape(B, -1)
        boxes = torch.cat((indices, boxes), dim=-1)

        roi_pooled = roi_align(frames, boxes, 2, spatial_scale=0.03125)
        roi_pooled = roi_pooled.reshape(B, -1)
        roi_pooled = self.roi_fc(roi_pooled)
        roi_pooled = self.roi_dropout(roi_pooled)
        roi_pooled = torch.relu(roi_pooled)

        data = torch.cat((spp_pooled, roi_pooled), dim=-1)
        data = torch.sigmoid(self.fc7(data))

        return data



class ProgressNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
