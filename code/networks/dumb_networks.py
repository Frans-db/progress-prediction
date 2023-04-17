import torch
import torch.nn as nn

class StaticNet(nn.Module):
    def __init__(self, device: torch.device, value: float = 0.5):
        super(StaticNet, self).__init__()
        # linear layer so optimizer & weight init work
        self.linear = nn.Linear(1, 1)
        self.device = device
        self.value = value

    def forward(self, x):
        B, S, _ = x.shape
        return torch.full((B, S), self.value, device=self.device, requires_grad=True)

class RandomNet(nn.Module):
    def __init__(self, device: torch.device):
        super(RandomNet, self).__init__()
        # linear layer so optimizer & weight init work
        self.linear = nn.Linear(1, 1)
        self.device = device

    def forward(self, x):
        B, S, _ = x.shape
        return torch.rand((B, S), self.value, device=self.device, requires_grad=True)