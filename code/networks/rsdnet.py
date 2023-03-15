import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class RSDNet(nn.Module):
    def __init__(self, p_dropout=0.3):
        super(RSDNet, self).__init__()
        self.lstm = nn.LSTM(2048, 64, 1)
        self.fc_rsd = nn.Linear(65, 1)
        self.fc_progress = nn.Linear(65, 1)

        self.cnn_dropout = nn.Dropout(p=p_dropout)
        self.lstm_dropout = nn.Dropout(p=p_dropout)

    def forward(self, data, elapsed, lengths):
        # TODO: Concat with t_elapsed (could probably just give rsd labels as input)
        
        batch_size, sequence_length, _ = data.shape
        num_samples = batch_size * sequence_length

        # packing & lstm
        data = self.cnn_dropout(data) # TODO: Check if a drop should happen here 
        data = data.reshape(batch_size, sequence_length, -1)
        packed = pack_padded_sequence(data, lengths, batch_first=True, enforce_sorted=False)
        packed, _ = self.lstm(packed)

        # unpacking
        unpacked, unpacked_lengths = pad_packed_sequence(packed, batch_first=True)
        unpacked = torch.cat((unpacked, elapsed.unsqueeze(dim=-1)), dim=-1)

        unpacked = unpacked.reshape(batch_size * sequence_length, -1)
        unpacked = self.lstm_dropout(unpacked)

        # rsd & progress predictions
        rsd_predictions = self.fc_rsd(unpacked)
        progress_predictions = torch.sigmoid(self.fc_progress(unpacked))

        rsd_predictions = rsd_predictions.reshape(batch_size, sequence_length)
        progress_predictions = progress_predictions.reshape(batch_size, sequence_length)

        return rsd_predictions, progress_predictions