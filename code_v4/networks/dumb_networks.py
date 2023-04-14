import torch
import torch.nn as nn

class StaticNet(nn.Module):
    def __init__(self, value: float, device: torch.device):
        super(StaticNet, self).__init__()
        self.linear = nn.Linear(1, 1)
        self.device = device
        self.value = value

    def forward(self, x):
        B, S, _ = x.shape
        progress = torch.full((B, S), self.value, device=self.device, requires_grad=True)
        return progress, progress, torch.full_like(x, self.value, device=self.device) 


class AverageNet(nn.Module):
    def __init__(self, value: float, device: torch.device):
        super(AverageNet, self).__init__()
        self.linear = nn.Linear(1, 1)
        self.device = device
        self.value = value

    def forward(self, x):
        B, S, _ = x.shape
        progress = torch.clamp(torch.arange(1, S+1, 1, dtype=torch.float32, device=self.device, requires_grad=True) / self.value, min=0, max=1)
        return progress.reshape(B, S), progress.reshape(B, S), x
