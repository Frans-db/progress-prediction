import torch
import torch.nn as nn

class SequentialLSTM(nn.Module):
    def __init__(self, data_embedding_size: int, forecasting_hidden_size: int, lstm_hidden_size: int, device: torch.device):
        super(SequentialLSTM, self).__init__()
        self.device = device
        self.data_embedding_size = data_embedding_size
        self.lstm_hidden_size = lstm_hidden_size

        self.forecasting_head = nn.Sequential(
            nn.Linear(self.data_embedding_size, forecasting_hidden_size),
            nn.Linear(forecasting_hidden_size, self.data_embedding_size)
        )
        proj_size = 1 if self.lstm_hidden_size > 1 else 0
        self.lstm = nn.LSTM(self.data_embedding_size, self.lstm_hidden_size,
                            1, batch_first=True, proj_size=proj_size)

    def forward(self, x):
        B, S, _ = x.shape

        # flatten x
        x = x.reshape(-1, self.data_embedding_size)
        # forecast representations
        forecasts = self.forecasting_head(x)
        # reshape x and forecasts
        x = x.reshape(1, -1, self.data_embedding_size)
        forecasts = forecasts.reshape(1, -1, self.data_embedding_size)

        progress = torch.zeros(B, S, 1, device=self.device)
        forecasted_progress = torch.full_like(progress, 0.5, device=self.device)
        # create hidden states
        hn, cn = (torch.zeros(1, 1, device=self.device), torch.zeros(1, self.lstm_hidden_size, device=self.device))
        for i in range(S): # unrolled LSTM
            # get current and forecasted representation
            item = x[:, i, :]
            forecasted_item = forecasts[:, i, :]
            # i th lstm pass, save hidden states
            item_progress, (hn, cn) = self.lstm(item, (hn, cn))
            # use i th lstm hidden states to forecast, discard hidden states
            forecasted_item_progress, _ = self.lstm(forecasted_item, (hn, cn))
            # store current and forecasted progress
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