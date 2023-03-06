import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class RSDNet(nn.Module):
    def __init__(self, embed_size=4069, p_dropout=0.5, finetune=False):
        super(RSDNet, self).__init__()
        self.resnet = models.resnet18()
        # remove last layer by replacing it with an identity
        self.resnet.fc = nn.Identity()
        self.lstm = nn.LSTM(512, 64, 1)
        self.fc1 = nn.Linear(64, 1)

    def forward(self, frames, boxes, lengths):
        batch_size, sequence_length, C, H, W = frames.shape
        num_samples = batch_size * sequence_length

        flat_frames = frames.reshape(num_samples, C, H, W)

        # resnet encoding
        encoded = self.resnet(flat_frames)
        encoded = encoded.reshape(batch_size, sequence_length, -1)

        # packing & lstm
        packed = pack_padded_sequence(encoded, lengths, batch_first=True, enforce_sorted=False)
        packed, _ = self.lstm(packed)

        # # unpacking & linear
        unpacked, unpacked_lengths = pad_packed_sequence(packed, batch_first=True)
        unpacked = unpacked.reshape(batch_size * sequence_length, -1)
        unpacked = self.fc1(unpacked)
        unpacked = unpacked.reshape(batch_size, sequence_length)

        # return unpacked
        return unpacked

class SimpleRSDNet(nn.Module):
    def __init__(self, embed_size=4069, p_dropout=0.5, finetune=False):
        super(SimpleRSDNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, 7)
        self.conv2 = nn.Conv2d(8, 16, 7)
        self.conv3 = nn.Conv2d(16, 32, 7)
        self.conv4 = nn.Conv2d(32, 32, 7)
        self.pool = nn.MaxPool2d(2, 2)
        # remove last layer by replacing it with an identity
        self.lstm = nn.LSTM(4032, 64, 1)
        self.fc1 = nn.Linear(64, 1)

    def forward(self, frames, boxes, lengths):
        batch_size, sequence_length, C, H, W = frames.shape
        num_samples = batch_size * sequence_length

        flat_frames = frames.reshape(num_samples, C, H, W)

        # encoding
        encoded = self.pool(self.conv1(flat_frames))
        encoded = self.pool(self.conv2(encoded))
        encoded = self.pool(self.conv3(encoded))
        encoded = self.pool(self.conv4(encoded))
        encoded = encoded.reshape(batch_size, sequence_length, -1)

        # packing & lstm
        packed = pack_padded_sequence(encoded, lengths, batch_first=True, enforce_sorted=False)
        packed, _ = self.lstm(packed)

        # # unpacking & linear
        unpacked, unpacked_lengths = pad_packed_sequence(packed, batch_first=True)
        unpacked = unpacked.reshape(batch_size * sequence_length, -1)
        unpacked = self.fc1(unpacked)
        unpacked = unpacked.reshape(batch_size, sequence_length)

        # return unpacked
        return unpacked