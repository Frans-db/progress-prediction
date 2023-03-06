import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class RSDNet(nn.Module):
    def __init__(self, basenet, p_dropout=0.3):
        super(RSDNet, self).__init__()
        self.resnet = basenet

        self.lstm = nn.LSTM(512, 64, 1)
        self.fc_rsd = nn.Linear(64, 1)
        self.fc_progress = nn.Linear(64, 1)

        self.cnn_dropout = nn.Dropout(p=p_dropout)
        self.lstm_dropout = nn.Dropout(p=p_dropout)

    def forward(self, frames, lengths):
        batch_size, sequence_length, C, H, W = frames.shape
        num_samples = batch_size * sequence_length

        flat_frames = frames.reshape(num_samples, C, H, W)

        # resnet encoding
        encoded = self.resnet(flat_frames)

        # packing & lstm
        encoded = self.cnn_dropout(encoded)
        encoded = encoded.reshape(batch_size, sequence_length, -1)
        packed = pack_padded_sequence(encoded, lengths, batch_first=True, enforce_sorted=False)
        packed, _ = self.lstm(packed)

        # unpacking
        unpacked, unpacked_lengths = pad_packed_sequence(packed, batch_first=True)
        unpacked = unpacked.reshape(batch_size * sequence_length, -1)
        unpacked = self.lstm_dropout(unpacked)

        # rsd & progress predictions
        rsd_predictions = self.fc_rsd(unpacked)
        progress_predictions = torch.sigmoid(self.fc_progress(unpacked))

        rsd_predictions = rsd_predictions.reshape(batch_size, sequence_length)
        progress_predictions = progress_predictions.reshape(batch_size, sequence_length)

        return rsd_predictions, progress_predictions