import torch
import torch.nn as nn
from torchvision.ops import roi_pool
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from pyramid_pooling import SpatialPyramidPooling

def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = get_device()

"""
Networks:
- LSTM, combine embeddings
- LSTM, separate embeddings
- Spatial, combine embeddings
- Spatial, separate embeddings
"""

class CombinedLSTM(nn.Module):
    pass

class SeparateDoubleLSTM(nn.Module):
    def __init__(self):
        super(SeparateDoubleLSTM, self).__init__()
        self.spp = SpatialPyramidPooling([2, 1])

        self.lstm_size = 4

        self.forecast_fc = nn.Linear(self.lstm_size, 15)
        self.embedding_fc = nn.Linear(15, 8)
        self.progress_fc = nn.Linear(self.lstm_size, 1)

        self.lstm = nn.LSTM(8, self.lstm_size, batch_first=True)

    def forward(self, frames):
        B, S, C, H, W = frames.shape
        num_samples = B * S

        flat_frames = frames.reshape(num_samples, C, H, W)
        pooled = self.spp(flat_frames)
        pooled = pooled.reshape(B, S, -1)

        forecasted_pooled = torch.zeros_like(pooled, device=device)
        progress = torch.zeros(B, S, 1, device=device)
        forecasted_progress = torch.zeros(B, S, 1, device=device)

        (hn, cn) = (torch.zeros(1, B, self.lstm_size, device=device), torch.zeros(1, B, self.lstm_size, device=device))
        for i in range(S):
            item = pooled[:, i, :]
   
            embedded = self.embedding_fc(item)
            embedded = embedded.unsqueeze(dim=1)

            rnn, (hn, cn) = self.lstm(embedded, (hn, cn))
            rnn = rnn.reshape(B, self.lstm_size)

            forecasted_item = self.forecast_fc(rnn)
            forecasted_pooled[:, i, :] = forecasted_item
            forecasted_embedded = self.embedding_fc(forecasted_item)
            forecasted_embedded = forecasted_embedded.unsqueeze(dim=1)

            forecasted_rnn, _ = self.lstm(forecasted_embedded, (hn, cn))


            forecasted_rnn = forecasted_rnn.reshape(B, self.lstm_size)

            item_progress = torch.sigmoid(self.progress_fc(rnn))
            forecasted_item_progress = torch.sigmoid(self.progress_fc(forecasted_rnn))

            progress[:, i] = item_progress
            forecasted_progress[:, i] = forecasted_item_progress

        progress = progress.reshape(B, S)
        forecasted_progress = forecasted_progress.reshape(B, S)

        return progress, forecasted_progress, pooled, forecasted_pooled

class SeparateSingleLSTM(nn.Module):
    def __init__(self):
        super(SeparateSingleLSTM, self).__init__()
        self.spp = SpatialPyramidPooling([2, 1])

        self.lstm_size = 4

        self.forecast_fc = nn.Linear(15, 15)
        self.embedding_fc = nn.Linear(15, 8)
        self.progress_fc = nn.Linear(self.lstm_size, 1)

        self.lstm = nn.LSTM(8, self.lstm_size, batch_first=True)

    def forward(self, frames):
        B, S, C, H, W = frames.shape
        num_samples = B * S

        flat_frames = frames.reshape(num_samples, C, H, W)
        pooled = self.spp(flat_frames)
        forecasted_pooled = self.forecast_fc(pooled)
        pooled = pooled.reshape(B, S, -1)
        forecasted_pooled = forecasted_pooled.reshape(B, S, -1)

        progress = torch.zeros(B, S, 1, device=device)
        forecasted_progress = torch.zeros(B, S, 1, device=device)

        (hn, cn) = (torch.zeros(1, B, self.lstm_size, device=device), torch.zeros(1, B, self.lstm_size, device=device))
        for i in range(S):
            item = pooled[:, i, :]
            forecasted_item = forecasted_pooled[:, i, :]
   
            embedded = self.embedding_fc(item)
            embedded = embedded.unsqueeze(dim=1)

            forecasted_embedded = self.embedding_fc(forecasted_item)
            forecasted_embedded = forecasted_embedded.unsqueeze(dim=1)

            rnn, (hn, cn) = self.lstm(embedded, (hn, cn))
            rnn = rnn.reshape(B, self.lstm_size)

            forecasted_rnn, _ = self.lstm(forecasted_embedded, (hn, cn))
            forecasted_rnn = forecasted_rnn.reshape(B, self.lstm_size)

            item_progress = torch.sigmoid(self.progress_fc(rnn))
            forecasted_item_progress = torch.sigmoid(self.progress_fc(forecasted_rnn))

            progress[:, i] = item_progress
            forecasted_progress[:, i] = forecasted_item_progress

        progress = progress.reshape(B, S)
        forecasted_progress = forecasted_progress.reshape(B, S)

        return progress, forecasted_progress, pooled, forecasted_pooled

class CombinedSpatial(nn.Module):
    pass


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