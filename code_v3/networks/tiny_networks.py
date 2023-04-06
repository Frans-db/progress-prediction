import torch
import torch.nn as nn
from torchvision.ops import roi_pool
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from pyramid_pooling import SpatialPyramidPooling

def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = get_device()

class TinyProgressNet(nn.Module):
    def __init__(self) -> int:
        super(TinyProgressNet, self).__init__()
        self.spp = SpatialPyramidPooling([2, 1])
        self.spp_fc = nn.Linear(30, 8)

        self.forecast_fc = nn.Linear(15, 15)

        self.fake_lstm = nn.Linear(8, 4)

        self.fc8 = nn.Linear(4, 1)

    def forward(self, frames: torch.FloatTensor) -> torch.FloatTensor:
        B, S, C, H, W = frames.shape
        num_samples = B * S
        flat_frames = frames.reshape(num_samples, C, H, W)

        # spp & linear
        pooled = self.spp(flat_frames)
        forecasted_pooled = self.forecast_fc(pooled)

        combined = torch.cat((pooled, forecasted_pooled), dim=-1)

        # linear & reshape
        embedded = torch.relu(self.spp_fc(combined))

        # lstm
        progress = self.fake_lstm(embedded)

        # progress
        progress = torch.sigmoid(self.fc8(progress))
        progress = progress.reshape(B, S)

        return progress, torch.full_like(progress, 0.5, device=device), pooled.reshape(B, S, -1), forecasted_pooled.reshape(B, S, -1)

class TinyLSTMNet(nn.Module):
    def __init__(self) -> int:
        super(TinyLSTMNet, self).__init__()
        self.spp = SpatialPyramidPooling([2, 1])
        self.spp_fc = nn.Linear(30, 8)

        self.forecast_fc = nn.Linear(15, 15)

        self.fake_lstm = nn.Linear(8, 4)

        self.fc8 = nn.Linear(4, 1)

    def forward(self, frames: torch.FloatTensor) -> torch.FloatTensor:
        B, S, C, H, W = frames.shape
        num_samples = B * S
        flat_frames = frames.reshape(num_samples, C, H, W)

        # spp & linear
        pooled = self.spp(flat_frames)
        forecasted_pooled = self.forecast_fc(pooled)

        combined = torch.cat((pooled, forecasted_pooled), dim=-1)

        # linear & reshape
        embedded = torch.relu(self.spp_fc(combined))

        # lstm
        progress = self.fake_lstm(embedded)

        # progress
        progress = torch.sigmoid(self.fc8(progress))
        progress = progress.reshape(B, S)

        return progress, torch.full_like(progress, 0.5, device=device), pooled.reshape(B, S, -1), forecasted_pooled.reshape(B, S, -1)

class OracleProgressNet(nn.Module):
    def __init__(self, delta_t: int) -> int:
        super(OracleProgressNet, self).__init__()
        self.delta_t = delta_t

        self.spp = SpatialPyramidPooling([4, 3, 2, 1])
        self.spp_fc = nn.Linear(180, 8)

        self.forecast_fc = nn.Linear(90, 90)

        self.fake_lstm = nn.Linear(8, 4)

        self.fc8 = nn.Linear(4, 1)

    def forward(self, frames: torch.FloatTensor) -> torch.FloatTensor:
        B, S, C, H, W = frames.shape
        num_samples = B * S
        flat_frames = frames.reshape(num_samples, C, H, W)

        # spp & linear
        pooled = self.spp(flat_frames).reshape(B, S, -1)
        forecasted_pooled = torch.ones_like(pooled, device=device)
        forecasted_pooled[:, :-self.delta_t, :] = pooled[:, self.delta_t:, :]

        pooled = pooled.reshape(num_samples, -1)
        forecasted_pooled = forecasted_pooled.reshape(num_samples, -1)

        combined = torch.cat((pooled, forecasted_pooled), dim=-1)

        # linear & reshape
        embedded = torch.relu(self.spp_fc(combined))

        # lstm
        progress = self.fake_lstm(embedded)

        # progress
        progress = torch.sigmoid(self.fc8(progress))
        progress = progress.reshape(B, S)

        return progress, torch.full_like(progress, 0.5, device=device), pooled.reshape(B, S, -1), forecasted_pooled.reshape(B, S, -1)