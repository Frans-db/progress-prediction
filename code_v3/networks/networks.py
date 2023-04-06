import torch
import torch.nn as nn
from torchvision.ops import roi_pool
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from pyramid_pooling import SpatialPyramidPooling

def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = get_device()


class ProgressNet(nn.Module):
    def __init__(self, embed_size: int = 4096, p_dropout: float = 0.5) -> int:
        super(ProgressNet, self).__init__()
        self.spp = SpatialPyramidPooling([4, 3, 2, 1])
        self.spp_fc = nn.Linear(90, embed_size)
        self.spp_dropout = nn.Dropout(p=p_dropout)

        self.fc7 = nn.Linear(embed_size, 64)
        self.forecast_fc = nn.Linear(embed_size, 64)

        self.lstm1 = nn.LSTM(64, 64, 1, batch_first=True)
        self.lstm2 = nn.LSTM(64, 32, 1, batch_first=True)

        self.fc8 = nn.Linear(32, 1)

    def forward(self, frames: torch.FloatTensor) -> torch.FloatTensor:
        B, S, C, H, W = frames.shape
        num_samples = B * S
        flat_frames = frames.reshape(num_samples, C, H, W)

        # spp & linear
        pooled = self.spp(flat_frames)
        pooled = self.spp_dropout(torch.relu(self.spp_fc(pooled)))
        # linear & reshape
        embedded = torch.relu(self.fc7(pooled))
        embedded = embedded.reshape(B, S, -1)
        # forecast
        forecasted_embedded = torch.relu(self.forecast_fc(pooled))
        forecasted_embedded = forecasted_embedded.reshape(B, S, -1)

        # lstm
        hidden1 = (torch.zeros(1, B, 64, device=device), torch.zeros(1, B, 64, device=device))
        hidden2 = (torch.zeros(1, B, 32, device=device), torch.zeros(1, B, 32, device=device))
        progress = torch.zeros(B, S, 32, device=device)
        forecasted_progress = torch.zeros(B, S, 32, device=device)
        for i in range(S):
            item = embedded[:, i, :].unsqueeze(dim=1)
            forecasted_item = forecasted_embedded[:, i, :].unsqueeze(dim=1)

            item, hidden1 = self.lstm1(item, hidden1)
            item, hidden2 = self.lstm2(item, hidden2)

            forecasted_item, _ = self.lstm1(forecasted_item, hidden1)
            forecasted_item, _ = self.lstm2(forecasted_item, hidden2)

            progress[:, i] = item.squeeze()
            forecasted_progress[:, i] = forecasted_item.squeeze()

        # progress
        progress = progress.reshape(num_samples, -1)
        progress = torch.sigmoid(self.fc8(progress))
        progress = progress.reshape(B, S)

        forecasted_progress = forecasted_progress.reshape(num_samples, -1)
        forecasted_progress = torch.sigmoid(self.fc8(forecasted_progress))
        forecasted_progress = forecasted_progress.reshape(B, S)

        return progress, forecasted_progress, embedded, forecasted_embedded

class PooledProgressNet(nn.Module):
    def __init__(self, embed_size: int = 4096, p_dropout: float = 0.5) -> int:
        super(PooledProgressNet, self).__init__()
        self.spp = SpatialPyramidPooling([4, 3, 2, 1])
        self.spp_fc = nn.Linear(90, embed_size)
        self.spp_dropout = nn.Dropout(p=p_dropout)

        self.forecast_fc = nn.Linear(90, 90)

        self.fc7 = nn.Linear(embed_size, 64)

        self.lstm1 = nn.LSTM(64, 64, 1, batch_first=True)
        self.lstm2 = nn.LSTM(64, 32, 1, batch_first=True)

        self.fc8 = nn.Linear(32, 1)

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
        hidden2 = (torch.zeros(1, B, 32, device=device), torch.zeros(1, B, 32, device=device))
        progress = torch.zeros(B, S, 32, device=device)
        forecasted_progress = torch.zeros(B, S, 32, device=device)
        for i in range(S):
            item = embedded[:, i, :].unsqueeze(dim=1)
            forecasted_item = forecasted_embedded[:, i, :].unsqueeze(dim=1)

            item, hidden1 = self.lstm1(item, hidden1)
            item, hidden2 = self.lstm2(item, hidden2)
            item = item.reshape(B, 32)

            forecasted_item, _ = self.lstm1(forecasted_item, hidden1)
            forecasted_item, _ = self.lstm2(forecasted_item, hidden2)
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

class RNNProgressNet(nn.Module):
    def __init__(self, embed_size: int = 4096, p_dropout: float = 0, num_heads: int = 1) -> int:
        super(RNNProgressNet, self).__init__()
        self.spp = SpatialPyramidPooling([4, 3, 2, 1])
        self.spp_fc = nn.Linear(90, embed_size)
        self.spp_dropout = nn.Dropout(p=p_dropout)

        self.forecast_fc = nn.Linear(32, 90)

        self.fc7 = nn.Linear(embed_size, 64)

        self.lstm1 = nn.LSTM(64, 64, 1, batch_first=True)
        self.lstm2 = nn.LSTM(64, 32, 1, batch_first=True)

        self.fc8 = nn.Linear(32, 1)

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
        forecasted_pooled = torch.zeros_like(pooled).reshape(B, S, -1)

        # lstm
        hidden1 = (torch.zeros(1, B, 64, device=device), torch.zeros(1, B, 64, device=device))
        hidden2 = (torch.zeros(1, B, 32, device=device), torch.zeros(1, B, 32, device=device))
        progress = torch.zeros(B, S, 32, device=device)
        forecasted_progress = torch.zeros(B, S, 32, device=device)
        for i in range(S):
            item = embedded[:, i, :].unsqueeze(dim=1)

            item, hidden1 = self.lstm1(item, hidden1)
            item, hidden2 = self.lstm2(item, hidden2)
            item = item.reshape(B, 32)

            forecasted_pool = self.forecast_fc(item)
            forecasted_pooled[:, i] = forecasted_pool
            forecasted_item = self.spp_dropout(torch.relu(self.spp_fc(forecasted_pool)))
            forecasted_item = torch.relu(self.fc7(forecasted_item))
            forecasted_item = forecasted_item.unsqueeze(dim=1)

            forecasted_item, _ = self.lstm1(forecasted_item, hidden1)
            forecasted_item, _ = self.lstm2(forecasted_item, hidden2)
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
