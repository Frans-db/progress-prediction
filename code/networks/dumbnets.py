import torch
import torch.nn as nn


class RandomNet(nn.Module):
    def __init__(self):
        super(ProgressNet, self).__init__()

    def forward(self, frames, boxes, lengths):
        batch_size, sequence_length, C, H, W = frames.shape
        
        return torch.rand(batch_size, sequence_length)


class StaticNet(nn.Module):
    def __init__(self, value: float = 0.5):
        super(ProgressNet, self).__init__()
        self.value = value

    def forward(self, frames, *):
        batch_size, sequence_length, C, H, W = frames.shape
        
        return torch.full((batch_size, sequence_length), self.value)


class RelativeNet(nn.Module):
    def __init__(self, value):
        super(ProgressNet, self).__init__()
        self.value = value

    def forward(self, indices):
        return indices / self.value