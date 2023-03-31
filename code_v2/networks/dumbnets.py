import torch
import torch.nn as nn


class RandomNet(nn.Module):
    def __init__(self, device):
        super(RandomNet, self).__init__()
        self.device = device

    def forward(self, frames, boxes, lengths):
        batch_size, sequence_length, C, H, W = frames.shape

        return torch.rand(batch_size, sequence_length, requires_grad=True).to(self.device)


class StaticNet(nn.Module):
    def __init__(self, device, value: float = 0.5):
        super(StaticNet, self).__init__()
        self.device = device
        self.value = value

    def forward(self, frames, *args, **kwargs):
        batch_size, sequence_length, C, H, W = frames.shape

        return torch.full((batch_size, sequence_length), self.value, requires_grad=True).to(self.device)


class RelativeNet(nn.Module):
    def __init__(self, device, value):
        super(RelativeNet, self).__init__()
        self.device = device
        self.value = value

    def forward(self, frames, *args, **kwargs):
        batch_size, sequence_length, C, H, W = frames.shape
        indices = torch.arange(start=0, end=sequence_length).repeat(batch_size, 1).to(self.device)

        return torch.clamp(indices / self.value, min=0, max=1)


class LSTMRelativeNet(nn.Module):
    def __init__(self, device, value):
        super(LSTMRelativeNet, self).__init__()
        self.device = device
        self.value = value
        self.lstm = nn.LSTM(1, 64, 1)
        self.fc1 = nn.Linear(64, 1)

    def forward(self, frames, *args, **kwargs):
        batch_size, sequence_length, C, H, W = frames.shape
        indices = torch.arange(start=0, end=sequence_length).float().to(self.device)
        indices = indices.repeat(batch_size, 1)
        indices = indices.reshape(batch_size, sequence_length, 1)

        outputs, _ = self.lstm(indices)
        outputs = outputs.reshape(batch_size * sequence_length, -1)
        outputs = self.fc1(outputs)
        outputs = outputs.reshape(batch_size, sequence_length)

        print(outputs.shape)

        return torch.sigmoid(outputs)
