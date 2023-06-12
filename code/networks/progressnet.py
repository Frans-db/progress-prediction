import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models
from torchvision.ops import roi_align
from typing import List

from .pyramidpooling import SpatialPyramidPooling


class ProgressNet(nn.Module):
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
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if backbone == "vgg16":
            self.backbone = models.vgg16().features
        else:
            raise Exception(f"Backbone {backbone} cannot be used for ProgressnetFlat")

        if backbone_path:
            self.backbone.load_state_dict(torch.load(backbone_path))

        # for param in self.backbone.parameters():
        #     param.requires_grad = False

        pooling_size = sum(map(lambda x: x**2, pooling_layers))
        self.roi_size = roi_size

        self.spp = SpatialPyramidPooling(pooling_layers)
        self.spp_fc = nn.Linear(512 * pooling_size, embed_dim)
        self.spp_dropout = nn.Dropout(p=dropout_chance)

        self.roi_fc = nn.Linear(512 * (roi_size**2), embed_dim)
        self.roi_dropout = nn.Dropout(p=dropout_chance)

        self.fc7 = nn.Linear(embed_dim * 2, 64)
        self.fc7_dropout = nn.Dropout(p=dropout_chance)

        self.lstm1 = nn.LSTM(64, 32, batch_first=True)
        self.lstm2 = nn.LSTM(32, 32, batch_first=True)

        self.fc8 = nn.Linear(32, 1)

    def forward(self, frames: torch.FloatTensor, boxes: torch.Tensor = None) -> torch.FloatTensor:
        B, S, C, H, W = frames.shape
        num_samples = B * S

        # print(boxes)
        # print(boxes.shape)

        if boxes is None:
            boxes = torch.FloatTensor([0, 0, 224, 224])
            boxes = boxes.repeat(num_samples, 1)
            boxes = boxes.to(self.device)

        frames = frames.reshape(num_samples, C, H, W)
        boxes = boxes.reshape(num_samples, 4)

        frames = self.backbone(frames)

        spp_pooled = self.spp(frames)
        spp_pooled = self.spp_fc(spp_pooled)
        spp_pooled = self.spp_dropout(spp_pooled)
        spp_pooled = torch.relu(spp_pooled)

        indices = torch.arange(0, num_samples, device=self.device).reshape(num_samples, -1)
        boxes = torch.cat((indices, boxes), dim=-1)

        roi_pooled = roi_align(frames, boxes, self.roi_size, spatial_scale=0.03125)
        roi_pooled = roi_pooled.reshape(num_samples, -1)
        roi_pooled = self.roi_fc(roi_pooled)
        roi_pooled = self.roi_dropout(roi_pooled)
        roi_pooled = torch.relu(roi_pooled)

        data = torch.cat((spp_pooled, roi_pooled), dim=-1)
        data = self.fc7(data)
        data = self.fc7_dropout(data)
        data = torch.relu(data)

        data = data.reshape(B, S, -1)
        data, _ = self.lstm1(data)
        data, _ = self.lstm2(data)
        data = data.reshape(num_samples, -1)
        data = self.fc8(data)
        data = torch.sigmoid(data)

        return data.reshape(B, S)

    def embed(self, frames, boxes = None):
        B, C, H, W = frames.shape

        if boxes is None:
            boxes = torch.FloatTensor([0, 0, 224, 224])
            boxes = boxes.repeat(B, 1)
            boxes = boxes.to(self.device)

        frames = self.backbone(frames)
        spp_pooled = self.spp(frames)
        spp_pooled = self.spp_fc(spp_pooled)
        spp_pooled = self.spp_dropout(spp_pooled)
        spp_pooled = torch.relu(spp_pooled)

        indices = torch.arange(0, B, device=self.device).reshape(B, -1)
        boxes = torch.cat((indices, boxes), dim=-1)

        roi_pooled = roi_align(frames, boxes, self.roi_size, spatial_scale=0.03125)
        roi_pooled = roi_pooled.reshape(B, -1)
        roi_pooled = self.roi_fc(roi_pooled)
        roi_pooled = self.roi_dropout(roi_pooled)
        roi_pooled = torch.relu(roi_pooled)

        return torch.cat((spp_pooled, roi_pooled), dim=-1)

# class ProgressNet(nn.Module):
#     def __init__(self, feature_dim: int, dropout_chance: float) -> None:
#         super().__init__()

#         self.fc7 = nn.Linear(feature_dim, 64)
#         self.fc7_dropout = nn.Dropout(p=dropout_chance)
#         self.lstm1 = nn.LSTM(64, 32, batch_first=True)
#         self.lstm2 = nn.LSTM(32, 32, batch_first=True)
#         self.fc8 = nn.Linear(32, 1)

#     def forward(self, data: torch.FloatTensor):
#         B, S, F = data.shape
#         data = torch.relu(self.fc7_dropout(self.fc7(data)))
#         data, _ = self.lstm1(data)
#         data, _ = self.lstm2(data)
#         data = torch.sigmoid(self.fc8(data))
#         return data.reshape(B, S)
