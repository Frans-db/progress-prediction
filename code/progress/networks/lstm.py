import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import resnet18, alexnet
from torchsummary import summary

class LSTMNetwork(nn.Module):
    def __init__(self, intermediary_representation_size: int = 18, hidden_size: int = 11, num_lstm_layers: int = 2) -> None:
        super().__init__()
        self.intermediary_representation_size = intermediary_representation_size
        self.hidden_size = hidden_size
        # Shared base network
        self.resnet = resnet18(num_classes=intermediary_representation_size)

        # LSTM progress head
        self.lstm = nn.LSTM(intermediary_representation_size, hidden_size, num_lstm_layers)
        self.fc1 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        batch_size, C, num_frames, H, W = x.shape
        # Shared base network
        x = x.reshape(batch_size * num_frames, C, H, W)
        x = self.resnet(x)
        x = x.reshape(batch_size, num_frames, self.intermediary_representation_size)
        # Calculate progress using LSTM
        y, (hn, cn) = self.lstm(x)
        y = y.reshape(batch_size * num_frames, self.hidden_size)
        y = self.fc1(y)
        y = y.reshape(batch_size, num_frames)

        return y

def main():
    net = resnet18(num_classes=10)
    summary(net, (3, 42, 42))

if __name__ == '__main__':
    main()