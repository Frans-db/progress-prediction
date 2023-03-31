import torch
import torch.nn as nn
from torchvision.ops import roi_pool
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from pyramid_pooling import SpatialPyramidPooling

def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = get_device()

class ProgressNet(nn.Module):
    def __init__(self, embed_size: int = 4096, p_dropout: float = 0, num_heads: int = 1) -> int:
        super(ProgressNet, self).__init__()
        self.num_heads = num_heads
        self.spp = SpatialPyramidPooling([4, 3, 2, 1])
        self.spp_fc = nn.Linear(90, embed_size)
        self.spp_dropout = nn.Dropout(p=p_dropout)

        self.fc7 = nn.Linear(embed_size, 64)
        # self.forecast_fc = nn.Linear(embed_size, 64)

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

        # lstm
        packed, _ = self.lstm1(embedded)
        packed, _ = self.lstm2(packed)

        # progress
        progress = torch.sigmoid(self.fc8(packed))
        progress = progress.reshape(B, S)

        return progress

    @staticmethod
    def init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.uniform_(m.bias, a=0, b=0)
        elif isinstance(m, nn.LSTM):
            nn.init.xavier_uniform_(m.weight_hh_l0)
            nn.init.xavier_uniform_(m.weight_ih_l0)
            nn.init.uniform_(m.bias_ih_l0, a=0, b=0)
            nn.init.uniform_(m.bias_hh_l0, a=0, b=0)

class UnrolledProgressNet(nn.Module):
    def __init__(self, embed_size: int = 4096, p_dropout: float = 0, num_heads: int = 1) -> int:
        super(UnrolledProgressNet, self).__init__()
        self.num_heads = num_heads
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
        hidden1 = (torch.zeros(1, 64, device=device), torch.zeros(1, 64, device=device))
        hidden2 = (torch.zeros(1, 32, device=device), torch.zeros(1, 32, device=device))
        progress = torch.zeros(B, S, 32, device=device)
        forecasted_progress = torch.zeros(B, S, 32, device=device)
        for i in range(S):
            item = embedded[:, i, :]
            forecasted_item = forecasted_embedded[:, i, :]

            item, hidden1 = self.lstm1(item, hidden1)
            item, hidden2 = self.lstm2(item, hidden2)

            forecasted_item, _ = self.lstm1(forecasted_item, hidden1)
            forecasted_item, _ = self.lstm2(forecasted_item, hidden2)

            progress[0, i] = item.squeeze()
            forecasted_progress[0, i] = forecasted_item.squeeze()

        # progress
        progress = progress.reshape(num_samples, -1)
        progress = torch.sigmoid(self.fc8(progress))
        progress = progress.reshape(B, S)

        forecasted_progress = forecasted_progress.reshape(num_samples, -1)
        forecasted_progress = torch.sigmoid(self.fc8(forecasted_progress))
        forecasted_progress = forecasted_progress.reshape(B, S)

        return progress, forecasted_progress