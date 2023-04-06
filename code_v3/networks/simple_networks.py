import torch
import torch.nn as nn
from torchvision.ops import roi_pool
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from pyramid_pooling import SpatialPyramidPooling

def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = get_device()


class SimpleProgressNet(nn.Module):
    def __init__(self, embed_size: int = 16, p_dropout: float = 0.5) -> int:
        super(SimpleProgressNet, self).__init__()
        self.spp = SpatialPyramidPooling([4, 3, 2, 1])
        self.spp_fc = nn.Linear(90, embed_size)
        self.spp_dropout = nn.Dropout(p=p_dropout)

        self.forecast_fc = nn.Linear(90, 90)

        self.fc7 = nn.Linear(embed_size, 8)

        self.lstm1 = nn.LSTM(8, 4, 1, batch_first=True)

        self.fc8 = nn.Linear(4, 1)

    def forward(self, frames: torch.FloatTensor) -> torch.FloatTensor:
        B, S, C, H, W = frames.shape
        num_samples = B * S
        flat_frames = frames.reshape(num_samples, C, H, W)

        # spp & linear
        pooled = self.spp(flat_frames)

        # linear & reshape
        embedded = self.spp_dropout(torch.relu(self.spp_fc(pooled)))
        embedded = torch.relu(self.fc7(embedded))
        embedded = embedded.reshape(B, S, -1)

        # forecast
        forecasted_pooled = self.forecast_fc(pooled)
        forecasted_embedded = self.spp_dropout(torch.relu(self.spp_fc(forecasted_pooled)))
        forecasted_embedded = torch.relu(self.fc7(forecasted_embedded))
        forecasted_embedded = forecasted_embedded.reshape(B, S, -1)

        # lstm
        hidden1 = (torch.zeros(1, B, 64, device=device), torch.zeros(1, B, 64, device=device))
        progress = torch.zeros(B, S, 32, device=device)
        forecasted_progress = torch.zeros(B, S, 32, device=device)
        for i in range(S):
            item = embedded[:, i, :].unsqueeze(dim=1)
            forecasted_item = forecasted_embedded[:, i, :].unsqueeze(dim=1)

            item, hidden1 = self.lstm1(item, hidden1)
            item = item.reshape(B, 32)

            forecasted_item, _ = self.lstm1(forecasted_item, hidden1)
            forecasted_item = forecasted_item.reshape(B, 32)

            progress[:, i] = item
            forecasted_progress[:, i] = forecasted_item

        # progress
        progress = progress.reshape(num_samples, -1)
        progress = torch.sigmoid(self.fc8(progress))
        progress = progress.reshape(B, S)

        forecasted_progress = forecasted_progress.reshape(num_samples, -1)
        forecasted_progress = torch.sigmoid(self.fc8(forecasted_progress))
        forecasted_progress = forecasted_progress.reshape(B, S)

        return progress, forecasted_progress, pooled.reshape(B, S, -1), forecasted_pooled.reshape(B, S, -1)

class SpatialProgressNet(nn.Module):
    def __init__(self, embed_size: int = 16, p_dropout: float = 0.5) -> int:
        super(SpatialProgressNet, self).__init__()
        self.spp = SpatialPyramidPooling([4, 3, 2, 1])
        self.spp_fc = nn.Linear(90, embed_size)
        self.spp_dropout = nn.Dropout(p=p_dropout)

        self.forecast_fc = nn.Linear(90, 90)

        self.fc7 = nn.Linear(embed_size, 8)

        self.fake_lstm = nn.Linear(8, 4)

        self.fc8 = nn.Linear(4, 1)

    def forward(self, frames: torch.FloatTensor) -> torch.FloatTensor:
        B, S, C, H, W = frames.shape
        num_samples = B * S
        flat_frames = frames.reshape(num_samples, C, H, W)

        # spp & linear
        pooled = self.spp(flat_frames)

        # linear & reshape
        embedded = self.spp_dropout(torch.relu(self.spp_fc(pooled)))
        embedded = torch.relu(self.fc7(embedded))

        # forecast
        forecasted_pooled = self.forecast_fc(pooled)
        forecasted_embedded = self.spp_dropout(torch.relu(self.spp_fc(forecasted_pooled)))
        forecasted_embedded = torch.relu(self.fc7(forecasted_embedded))

        # lstm
        hidden1 = (torch.zeros(1, B, 64, device=device), torch.zeros(1, B, 64, device=device))
        hidden2 = (torch.zeros(1, B, 32, device=device), torch.zeros(1, B, 32, device=device))
        progress = torch.zeros(B, S, 32, device=device)
        forecasted_progress = torch.zeros(B, S, 32, device=device)

        progress = self.fake_lstm(embedded)
        forecasted_progress = self.fake_lstm(forecasted_embedded)

        # progress
        progress = torch.sigmoid(self.fc8(progress))
        progress = progress.reshape(B, S)

        forecasted_progress = torch.sigmoid(self.fc8(forecasted_progress))
        forecasted_progress = forecasted_progress.reshape(B, S)

        return progress, forecasted_progress, pooled.reshape(B, S, -1), forecasted_pooled.reshape(B, S, -1)

class WeirdProgressNet(nn.Module):
    def __init__(self, embed_size: int = 16, p_dropout: float = 0.5, delta_t: int = 10) -> int:
        super(WeirdProgressNet, self).__init__()
        self.delta_t = delta_t

        self.spp = SpatialPyramidPooling([4, 3, 2, 1])
        self.spp_fc = nn.Linear(180, embed_size)
        self.spp_dropout = nn.Dropout(p=p_dropout)

        self.forecast_fc = nn.Linear(180, 90)

        self.fc7 = nn.Linear(embed_size, 8)

        self.fake_lstm = nn.Linear(8, 4)

        self.fc8 = nn.Linear(4, 1)

    def forward(self, frames: torch.FloatTensor) -> torch.FloatTensor:
        B, S, C, H, W = frames.shape
        num_samples = B * S
        flat_frames = frames.reshape(num_samples, C, H, W)

        # spp & linear
        pooled = self.spp(flat_frames)
        pooled = pooled.reshape(B, S, -1)
        pooled_shape = (pooled.shape[0], pooled.shape[1] + self.delta_t, pooled.shape[2])
        forecasted_pooled = torch.zeros(pooled_shape).to(device)

        all_progress = torch.zeros(B, S).to(device)
        all_forecasted_progress = torch.zeros(B, S).to(device)
        for i in range(S):
            item = pooled[:, i, :]
            forecasted_item = forecasted_pooled[:, i, :]
            combined = torch.cat((item, forecasted_item), dim=-1)

            forecasted_pool = self.forecast_fc(combined)
            forecasted_pooled[:, i+self.delta_t, :] = forecasted_pool
            
            embedded = self.spp_dropout(torch.relu(self.spp_fc(combined)))
            embedded = self.fc7(embedded)

            progress = self.fake_lstm(embedded)
            forecasted_progress = self.fake_lstm(embedded)

            progress = torch.sigmoid(self.fc8(progress))
            forecasted_progress = torch.sigmoid(self.fc8(forecasted_progress))

            all_progress[:, i] = progress
            all_forecasted_progress[:, i] = forecasted_progress

        return all_progress, all_forecasted_progress, pooled, forecasted_pooled[:, self.delta_t:]

class WeirderProgressNet(nn.Module):
    def __init__(self, embed_size: int = 16, p_dropout: float = 0.5, delta_t: int = 10) -> int:
        super(WeirderProgressNet, self).__init__()
        self.delta_t = delta_t

        self.spp = SpatialPyramidPooling([4, 3, 2, 1])
        self.spp_fc = nn.Linear(180, embed_size)
        self.spp_dropout = nn.Dropout(p=p_dropout)

        self.forecast_fc = nn.Linear(90, 90)

        self.fc7 = nn.Linear(embed_size, 8)

        self.fake_lstm = nn.Linear(8, 4)

        self.fc8 = nn.Linear(4, 1)

    def forward(self, frames: torch.FloatTensor) -> torch.FloatTensor:
        B, S, C, H, W = frames.shape
        num_samples = B * S
        flat_frames = frames.reshape(num_samples, C, H, W)

        # spp & linear
        pooled = self.spp(flat_frames)
        pooled = pooled.reshape(B, S, -1)
        pooled_shape = (pooled.shape[0], pooled.shape[1] + self.delta_t, pooled.shape[2])
        forecasted_pooled = torch.zeros(pooled_shape).to(device)

        all_progress = torch.zeros(B, S).to(device)
        all_forecasted_progress = torch.zeros(B, S).to(device)
        for i in range(S):
            item = pooled[:, i, :]
            
            forecasted_item = self.forecast_fc(item)
            forecasted_pooled[:, i+self.delta_t, :] = forecasted_item
            combined = torch.cat((item, forecasted_item), dim=-1)

            embedded = self.spp_dropout(torch.relu(self.spp_fc(combined)))
            embedded = self.fc7(embedded)

            progress = self.fake_lstm(embedded)
            forecasted_progress = self.fake_lstm(embedded)

            progress = torch.sigmoid(self.fc8(progress))
            forecasted_progress = torch.sigmoid(self.fc8(forecasted_progress))

            all_progress[:, i] = progress
            all_forecasted_progress[:, i] = forecasted_progress

        return all_progress, all_forecasted_progress, pooled, forecasted_pooled[:, self.delta_t:]