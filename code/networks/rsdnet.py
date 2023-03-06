import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class RSDNet(nn.Module):
    def __init__(self, p_dropout=0.3, finetune=False):
        super(RSDNet, self).__init__()
        self.finetune = finetune
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.resnet.fc = nn.Sequential(
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

        self.lstm = nn.LSTM(512, 64, 1)
        self.fc_rsd = nn.Linear(64, 1)
        self.fc_progress = nn.Linear(64, 1)

    def disable_finetune(self):
        self.resnet.fc = nn.Identity()

    def forward(self, frames, lengths):
        batch_size, sequence_length, C, H, W = frames.shape
        num_samples = batch_size * sequence_length

        flat_frames = frames.reshape(num_samples, C, H, W)

        # resnet encoding
        encoded = self.resnet(flat_frames)

        if self.finetune:
            progress = encoded.reshape(batch_size, sequence_length)
            return progress.clone(), progress
        else:
            # packing & lstm
            encoded = encoded.reshape(batch_size, sequence_length, -1)
            packed = pack_padded_sequence(encoded, lengths, batch_first=True, enforce_sorted=False)
            packed, _ = self.lstm(packed)

            # unpacking
            unpacked, unpacked_lengths = pad_packed_sequence(packed, batch_first=True)
            unpacked = unpacked.reshape(batch_size * sequence_length, -1)

            # rsd & progress predictions
            rsd_predictions = self.fc_rsd(unpacked)
            progress_predictions = torch.sigmoid(self.fc_progress(unpacked))

            rsd_predictions = rsd_predictions.reshape(batch_size, sequence_length)
            progress_predictions = progress_predictions.reshape(batch_size, sequence_length)

            return rsd_predictions, progress_predictions