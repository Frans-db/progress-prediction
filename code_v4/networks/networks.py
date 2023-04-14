import torch
import torch.nn as nn

class ProgressNet2D(nn.Module):
    def __init__(self, data_embedding_size: int,  device: torch.device):
        super(ProgressNet2D, self).__init__()
        self.device = device

        self.fc7 = nn.Linear(data_embedding_size, 64)

        self.lstm1 = nn.Linear(64, 64)
        self.lstm2 = nn.Linear(64, 32)

        self.fc8 = nn.Linear(32, 1)


    def forward(self, x):
        B, S, _ = x.shape
        forecasts = torch.rand_like(x, device=self.device)
        
        # x = torch.ones(B, S, dtype=torch.float32, device=self.device, requires_grad=True)
        # x = torch.rand((B, S), dtype=torch.float32, device=self.device, requires_grad=True)

        y = torch.arange(1, S+1, 1, dtype=torch.float32, device=self.device, requires_grad=True) / 2113.340410958904
        y = y.reshape(B*S, 1)
        x = x.reshape(B*S, -1)
        x = torch.concat((x, y), dim=-1)

        x = torch.relu(self.fc7(x))
        x = self.lstm1(x)
        x = self.lstm2(x)
        x = torch.relu(x)
        x = torch.sigmoid(self.fc8(x))
        x = x.reshape(B, S)

        return x, torch.full_like(x, 0.5, device=self.device), forecasts

class ProgressNet(nn.Module):
    def __init__(self, data_embedding_size: int,  device: torch.device):
        super(ProgressNet, self).__init__()
        self.device = device

        self.fc7 = nn.Linear(data_embedding_size, 64)

        self.lstm1 = nn.LSTM(64, 64, 1, batch_first=True)
        self.lstm2 = nn.LSTM(64, 32, 1, batch_first=True)

        self.fc8 = nn.Linear(32, 1)


    def forward(self, x):
        B, S, _ = x.shape
        forecasts = torch.rand_like(x, device=self.device)
        
        # x = torch.ones(B, S, dtype=torch.float32, device=self.device, requires_grad=True)
        # x = torch.arange(1, S+1, 1, dtype=torch.float32, device=self.device, requires_grad=True) / 2113.340410958904
        # x = torch.rand((B, S), dtype=torch.float32, device=self.device, requires_grad=True)

        x = x.reshape(B*S, -1)
        x = torch.relu(self.fc7(x))

        x = x.reshape(B, S, -1)
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x = torch.relu(x.reshape(B*S, -1))

        x = torch.sigmoid(self.fc8(x))
        x = x.reshape(B, S)

        return x, torch.full_like(x, 0.5, device=self.device), forecasts
        
