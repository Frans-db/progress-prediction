import torch
import torch.nn as nn
from torchvision.ops import roi_pool
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from .layers import SpatialPyramidPooling

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.uniform_(m.bias, a=0, b=0)
    elif isinstance(m, nn.LSTM):
        nn.init.xavier_uniform_(m.weight_hh_l0)
        nn.init.xavier_uniform_(m.weight_ih_l0)
        nn.init.uniform_(m.bias_ih_l0, a=0, b=0)
        nn.init.uniform_(m.bias_hh_l0, a=0, b=0)


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

        init_weights(self.spp_fc)
        init_weights(self.roi_fc)
        init_weights(self.fc7)

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


class RNNHead(nn.Module):
    def __init__(self):
        super(RNNHead, self).__init__()
        self.lstm1 = nn.LSTM(64, 64, 1)
        self.lstm2 = nn.LSTM(64, 32, 1)
        init_weights(self.lstm1)
        init_weights(self.lstm2)

    def forward(self, embedding, lengths):
        batch_size, sequence_length, _ = embedding.shape

        # packing & lstm
        packed = pack_padded_sequence(embedding, lengths, batch_first=True, enforce_sorted=False)
        packed, lstm1_memory = self.lstm1(packed)
        packed, lstm2_memory = self.lstm2(packed)
        
        # unpacking
        unpacked, unpacked_lengths = pad_packed_sequence(packed, batch_first=True)

        return unpacked

class ForecastingHead(nn.Module):
    def __init__(self):
        super(ForecastingHead, self).__init__()
        self.fc1 = nn.Linear(32, 64)
        self.fc2 = nn.Linear(64, 256)
        self.fc3 = nn.Linear(256, 64)
        self.fc4 = nn.Linear(64, 32)

    def forward(self, x):
        batch_size, sequence_length, _ = x.shape
        x = x.reshape(batch_size * sequence_length, -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        return x.reshape(batch_size, sequence_length, -1)

class ProgressHead(nn.Module):
    def __init__(self):
        super(ProgressHead, self).__init__()
        self.fc8 = nn.Linear(32, 1)
        init_weights(self.fc8)

    def forward(self, concatenated):
        batch_size, sequence_length, _ = concatenated.shape

        concatenated = concatenated.reshape(batch_size*sequence_length, -1)
        predictions = torch.sigmoid(self.fc8(concatenated))
        predictions = predictions.reshape(batch_size, sequence_length)

        return predictions


class ProgressForecastingNet(nn.Module):
    def __init__(self, device, delta_t: int, embed_size=2048, p_dropout=0):
        super(ProgressForecastingNet, self).__init__()
        self.device = device
        self.delta_t = delta_t

        self.embedding = EmbeddingHead(device, embed_size=embed_size, p_dropout=p_dropout)
        self.rnn = RNNHead()
        self.forecasting = ForecastingHead()
        self.progress = ProgressHead()

    def forward(self, frames, boxes, lengths):
        batch_size, sequence_length, C, H, W = frames.shape

        embeddings = self.embedding(frames, boxes)
        rnn_embeddings = self.rnn(embeddings, lengths)

        forecasted_embeddings = self.forecasting(rnn_embeddings)
        progress_predictions = self.progress(rnn_embeddings)
        future_progress_predictions = self.progress(forecasted_embeddings)

        return rnn_embeddings, progress_predictions, forecasted_embeddings, future_progress_predictions

    def finetune(self):
        for name, param in self.named_parameters():
            if 'fc8' in name:
                continue
            param.requires_grad = False
