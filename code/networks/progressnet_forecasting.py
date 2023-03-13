import torch
import torch.nn as nn
from torchvision.ops import roi_pool
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from .layers import SpatialPyramidPooling


class EmbeddingHead(nn.Module):
    def __init__(self, device, embed_size=2048, p_dropout=0):
        super(EmbeddingHead, self).__init__()
        self.device = device
        self.spp = SpatialPyramidPooling([4, 3, 2, 1])
        self.spp_fc = nn.Linear(90, embed_size)
        self.spp_dropout = nn.Dropout(p=p_dropout)

        self.roi_fc = nn.Linear(3*7*7, embed_size)
        self.roi_dropout = nn.Dropout(p=p_dropout)

        self.fc7 = nn.Linear(embed_size*2, 64)
        self.fc7_dropout = nn.Dropout(p=p_dropout)

        self._init_weights(self.spp_fc)
        self._init_weights(self.roi_fc)
        self._init_weights(self.fc7)

    def forward(self, frames, boxes):
        batch_size, sequence_length, C, H, W = frames.shape
        num_samples = batch_size * sequence_length

        flat_frames = frames.reshape(num_samples, C, H, W)
        flat_boxes = boxes.reshape(num_samples, 4)
        indices = torch.arange(start=0, end=num_samples).reshape(
            num_samples, 1).to(self.device)

        boxes_with_indices = torch.cat((indices, flat_boxes), dim=-1)

        # spp & linear
        pooled = self.spp(flat_frames)
        pooled = self.spp_dropout(torch.relu(self.spp_fc(pooled)))

        # roi & linear
        roi = roi_pool(flat_frames, boxes_with_indices, 7)
        roi = torch.flatten(roi, start_dim=1)
        roi = self.roi_dropout(torch.relu(self.roi_fc(roi)))

        # linear & reshape
        concatenated = torch.cat((pooled, roi), dim=-1)
        concatenated = self.fc7_dropout(torch.relu(self.fc7(concatenated)))
        concatenated = concatenated.reshape(batch_size, sequence_length, -1)

        return concatenated

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.uniform_(m.bias, a=0, b=0)
        elif isinstance(m, nn.LSTM):
            nn.init.xavier_uniform_(m.weight_hh_l0)
            nn.init.xavier_uniform_(m.weight_ih_l0)
            nn.init.uniform_(m.bias_ih_l0, a=0, b=0)
            nn.init.uniform_(m.bias_hh_l0, a=0, b=0)


class ForecastingHead(nn.Module):
    def __init__(self, embed_size=2048):
        super(ForecastingHead, self).__init__()
        self.embed_size = embed_size
        # go from 64 to embed size (2048)
        self.fc1 = nn.Linear(64, 768)
        self.fc2 = nn.Linear(768, 1472)
        self.fc3 = nn.Linear(1472, self.embed_size)

    def forward(self, x):
        batch_size, sequence_length, _ = x.shape
        x = x.reshape(batch_size * sequence_length, -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return x.reshape(batch_size, sequence_length, -1)

class ProgressHead(nn.Module):
    def __init__(self):
        super(ProgressHead, self).__init__()
        self.lstm1 = nn.LSTM(64, 64, 1)
        self.lstm2 = nn.LSTM(64, 32, 1)
        self.fc8 = nn.Linear(32, 1)

        self._init_weights(self.lstm1)
        self._init_weights(self.lstm2)
        self._init_weights(self.fc8)

    def forward(self, concatenated, lengths):
        batch_size, sequence_length, _ = concatenated.shape

        # packing & lstm
        packed = pack_padded_sequence(
            concatenated, lengths, batch_first=True, enforce_sorted=False)
        packed, lstm1_memory = self.lstm1(packed)
        packed, lstm2_memory = self.lstm2(packed)

        # unpacking
        unpacked, unpacked_lengths = pad_packed_sequence(packed, batch_first=True)
        unpacked = unpacked.reshape(batch_size * sequence_length, -1)

        # progress heads
        predictions = torch.sigmoid(self.fc8(unpacked))
        predictions = predictions.reshape(batch_size, sequence_length)

        return predictions

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.uniform_(m.bias, a=0, b=0)
        elif isinstance(m, nn.LSTM):
            nn.init.xavier_uniform_(m.weight_hh_l0)
            nn.init.xavier_uniform_(m.weight_ih_l0)
            nn.init.uniform_(m.bias_ih_l0, a=0, b=0)
            nn.init.uniform_(m.bias_hh_l0, a=0, b=0)


class ProgressForecastingNet(nn.Module):
    def __init__(self, device, embed_size=2048, p_dropout=0):
        super(ProgressForecastingNet, self).__init__()
        self.device = device

        self.embedding = EmbeddingHead(device, embed_size=embed_size, p_dropout=p_dropout)
        self.forecasting = ForecastingHead(embed_size=embed_size)
        self.progress = ProgressHead()

    def forward(self, frames, boxes, lengths):
        batch_size, sequence_length, C, H, W = frames.shape

        embeddings = self.embedding(frames, boxes)

        # forecasted_embeddings = self.forecasting(embeddings)
        progress_predictions = self.progress(embeddings, lengths)

        # print(self.progress.lstm1)

        return progress_predictions

    def finetune(self):
        for name, param in self.named_parameters():
            if 'fc8' in name:
                continue
            param.requires_grad = False
