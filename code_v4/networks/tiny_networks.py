import torch
import torch.nn as nn

class SequentialLSTM(nn.Module):
    def __init__(self, data_embedding_size: int, forecasting_hidden_size: int, lstm_hidden_size: int, device: torch.device):
        super(SequentialLSTM, self).__init__()
        self.device = device
        self.lstm_hidden_size = lstm_hidden_size

        self.forecasting_head = nn.Sequential(
            nn.Linear(15, forecasting_hidden_size),
            nn.Linear(forecasting_hidden_size, 15)
        )
        proj_size = 1 if self.lstm_hidden_size > 1 else 0
        self.lstm = nn.LSTM(data_embedding_size, self.lstm_hidden_size,
                            1, batch_first=True, proj_size=proj_size)

    def forward(self, x):
        B, S, _ = x.shape

        forecasts = self.forecasting_head(x.reshape(-1, 15)).reshape(1, -1, 15)

        progress = torch.zeros(B, S, 1, device=self.device)
        forecasted_progress = torch.zeros_like(progress, device=self.device)
        hn, cn = (torch.zeros(1, 1, device=self.device), torch.zeros(1, self.lstm_hidden_size, device=self.device))
        for i in range(S):
            item = x[:, i, :]
            forecasted_item = forecasts[:, i, :]

            item_progress, (hn, cn) = self.lstm(item, (hn, cn))
            forecasted_item_progress, _ = self.lstm(forecasted_item, (hn, cn))

            progress[:, i, :] = item_progress
            forecasted_progress[:, i, :] = forecasted_item_progress

        return progress.reshape(B, S), forecasted_progress.reshape(B, S), forecasts


class ParallelLSTM(nn.Module):
    def __init__(self, data_embedding_size: int, forecasting_hidden_size: int, lstm_hidden_size: int, device: torch.device):
        super(ParallelLSTM, self).__init__()
        self.device = device
        self.lstm_hidden_size = lstm_hidden_size

        self.forecasting_head = nn.Sequential(
            nn.Linear(15, forecasting_hidden_size),
            nn.Linear(forecasting_hidden_size, 15)
        )
        proj_size = 1 if self.lstm_hidden_size > 1 else 0
        self.lstm = nn.LSTM(data_embedding_size*2, self.lstm_hidden_size,
                            1, batch_first=True, proj_size=proj_size)

    def forward(self, x):
        B, S, _ = x.shape

        forecasts = self.forecasting_head(x.reshape(-1, 15)).reshape(1, -1, 15)
        x = torch.cat((x, forecasts), dim=-1)

        progress = torch.zeros(B, S, 1, device=self.device)
        forecasted_progress = torch.zeros_like(progress, device=self.device)
        hn, cn = (torch.zeros(1, 1, device=self.device), torch.zeros(1, self.lstm_hidden_size, device=self.device))
        for i in range(S):
            item = x[:, i, :]
            item_progress, (hn, cn) = self.lstm(item, (hn, cn))
            progress[:, i, :] = item_progress

        return progress.reshape(B, S), forecasted_progress.reshape(B, S), forecasts