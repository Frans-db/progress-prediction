import torch
import torch.nn as nn
from torchvision.ops import roi_pool

from .pyramid_pooling import SpatialPyramidPooling


class ProgressNet(nn.Module):
    def __init__(self, embedding_size: int, p_dropout: float) -> None:
        super(ProgressNet, self).__init__()
        self.spp = SpatialPyramidPooling([1, 2, 3, 4])
        self.spp_fc = nn.Linear(90, embedding_size)
        self.spp_dropout = nn.Dropout(p=p_dropout)

        self.fc7 = nn.Linear(embedding_size, 64)
        self.fc7_dropout = nn.Dropout(p=p_dropout)

        self.lstm1 = nn.LSTM(64, 64, 1, batch_first=True)
        self.lstm2 = nn.LSTM(64, 32, 1, batch_first=True)

        self.fc8 = nn.Linear(32, 1)

    def forward(self, frames: torch.FloatTensor) -> torch.FloatTensor:
        B, S, C, H, W = x.shape
        num_samples = B * S

        flat_frames = frames.reshape(num_samples, C, H, W)

        pooled = self.spp(flat_frames)
        pooled = self.spp_fc(pooled)
        pooled = self.spp_dropout(pooled)
        pooled = torch.relu(pooled)

        pooled = self.fc7(pooled)
        pooled = self.fc7_dropout(pooled)
        pooled = torch.relu(pooled)

        rnn = pooled.reshape(B, S, -1)
        rnn, _ = self.lstm1(rnn)
        rnn, _ = self.lstm2(rnn)
        rnn = torch.relu(rnn)

        progress = rnn.reshape(num_samples, -1)
        progress = self.fc8(progress)
        progress = torch.sigmoid(progress)

        return progress


class ProgressNetFeatures(nn.Module):
    def __init__(self, embedding_size: int, p_dropout: float) -> None:
        super(ProgressNetFeatures, self).__init__()
        self.fc7 = nn.Linear(embedding_size, 64)
        self.fc7_dropout = nn.Dropout(p=p_dropout)
        self.lstm1 = nn.LSTM(64, 64, 1, batch_first=True)
        self.lstm2 = nn.LSTM(64, 32, 1, batch_first=True)
        self.fc8 = nn.Linear(32, 1)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        B, S, _ = x.shape
        num_samples = B * S

        x = x.reshape(num_samples, -1)
        x = self.fc7(x)
        x = self.fc7_dropout(x)
        x = torch.relu(x)

        x = x.reshape(B, S, -1)
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x = x.reshape(num_samples, -1)
        x = torch.relu(x)

        x = self.fc8(x)
        x = torch.sigmoid(x)
        x = x.reshape(B, S)

        return x


class ProgressNetBoundingBoxes(nn.Module):
    def __init__(self, embedding_size: int, p_dropout: float, device: torch.device) -> None:
        super(ProgressNetBoundingBoxes, self).__init__()
        self.device = device

        self.spp = SpatialPyramidPooling([1, 2, 3, 4])
        self.spp_fc = nn.Linear(90, embedding_size)
        self.spp_dropout = nn.Dropout(p=p_dropout)

        self.roi_fc = nn.Linear(3*7*7, embedding_size)
        self.roi_dropout = nn.Dropout(p=p_dropout)

        self.fc7 = nn.Linear(embedding_size*2, 64)
        self.fc7_dropout = nn.Dropout(p=p_dropout)

        self.lstm1 = nn.LSTM(64, 64, 1, batch_first=True)
        self.lstm2 = nn.LSTM(64, 32, 1, batch_first=True)

        self.fc8 = nn.Linear(32, 1)

    def forward(self, frames: torch.FloatTensor, boxes: torch.FloatTensor) -> torch.FloatTensor:
        B, S, C, H, W = x.shape
        num_samples = B * S

        flat_frames = frames.reshape(num_samples, C, H, W)
        flat_boxes = boxes.reshape(num_samples, 4)
        box_indices = otrch.arange(start=0, end=num_samples).reshape(
            num_samples, 1).to(self.device)

        boxes_with_indices = torch.cat((indices, flat_boxes), dim=-1)

        pooled = self.spp(flat_frames)
        pooled = self.spp_fc(pooled)
        pooled = self.spp_dropout(pooled)
        pooled = torch.relu(pooled)

        roi = roi_pool(flat_frames, boxes_with_indices, 7)
        roi = roi.reshape(num_samples, -1)
        roi = self.roi_fc(roi)
        roi = self.roi_dropout(roi)
        roi = torch.relu(roi)

        concatenated = torch.cat((pooled, roi), dim=-1)
        concatenated = self.fc7(concatenated)
        concatenated = self.fc7_dropout(concatenated)
        concatenated = torch.relu(concatenated)

        rnn = concatenated.reshape(B, S, -1)
        rnn, _ = self.lstm1(rnn)
        rnn, _ = self.lstm2(rnn)
        rnn = torch.relu(rnn)

        progress = rnn.reshape(num_samples, -1)
        progress = self.fc8(progress)
        progress = torch.sigmoid(progress)

        return progress
