import torch
import torch.nn as nn

class StaticNet(nn.Module):
    def __init__(self, device: torch.device, value: float = 0.5):
        super(StaticNet, self).__init__()
        # linear layer so optimizer & weight init work
        self.linear = nn.Linear(1, 1)
        self.device = device
        self.value = value

    def forward(self, x, *args, **kwargs):
        B, S = x.shape[0], x.shape[1]
        return torch.full((B, S), self.value, device=self.device, requires_grad=True)

class RandomNet(nn.Module):
    def __init__(self, device: torch.device):
        super(RandomNet, self).__init__()
        # linear layer so optimizer & weight init work
        self.linear = nn.Linear(1, 1)
        self.device = device

    def forward(self, x, *args, **kwargs):
        B, S = x.shape[0], x.shape[1]
        return torch.rand((B, S), device=self.device, requires_grad=True)