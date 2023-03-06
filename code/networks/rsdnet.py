import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class RSDNet(nn.Module):
    def __init__(self, p_dropout=0.3, finetune=False):
        super(RSDNet, self).__init__()
        self.finetune = finetune
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        if self.finetune:
            # Replace last layer of resnet with a progress prediction head
            self.resnet.fc = nn.Sequential(
                nn.Linear(512, 1),
                nn.Sigmoid()
            )
        else:
            # Remove last layer of resnet, second to last layer is used as an embedding
            self.resnet.fc = nn.Identity()

        self.lstm = nn.LSTM(512, 64, 1)
        self.fc_rsd = nn.Linear(64, 1)
        self.fc_progress = nn.Linear(64, 1)

    def forward(self, frames, lengths):
        batch_size, sequence_length, C, H, W = frames.shape
        num_samples = batch_size * sequence_length

        flat_frames = frames.reshape(num_samples, C, H, W)

        # resnet encoding
        encoded = self.resnet(flat_frames)
        encoded = encoded.reshape(batch_size, sequence_length, -1)

        if self.finetune:
            print(encoded.shape)
            exit(0)
        else:
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