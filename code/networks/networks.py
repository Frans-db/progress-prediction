import torch
import torch.nn as nn
from torchvision.ops import roi_pool
import torchvision.models as models

from .pyramid_pooling import SpatialPyramidPooling


class ProgressNet(nn.Module):
    def __init__(self, embedding_size: int, p_dropout: float) -> None:
        super(ProgressNet, self).__init__()
        self.spp = SpatialPyramidPooling([1, 2, 3, 4])
        self.spp_fc = nn.Linear(90, embedding_size)
        self.spp_dropout = nn.Dropout(p=p_dropout)

        self.fc7 = nn.Linear(embedding_size, 64)
        self.fc7_dropout = nn.Dropout(p=p_dropout)

        self.lstm1 = nn.LSTM(64, 64, 1, batch_first=True)
        self.lstm2 = nn.LSTM(64, 32, 1, batch_first=True)

        self.fc8 = nn.Linear(32, 1)

    def forward(self, frames: torch.FloatTensor) -> torch.FloatTensor:
        B, S, C, H, W = frames.shape
        num_samples = B * S

        flat_frames = frames.reshape(num_samples, C, H, W)

        pooled = self.spp(flat_frames)
        pooled = self.spp_fc(pooled)
        pooled = self.spp_dropout(pooled)
        pooled = torch.relu(pooled)

        pooled = self.fc7(pooled)
        pooled = self.fc7_dropout(pooled)
        pooled = torch.relu(pooled)

        rnn = pooled.reshape(B, S, -1)
        rnn, _ = self.lstm1(rnn)
        rnn, _ = self.lstm2(rnn)
        rnn = torch.relu(rnn)

        progress = rnn.reshape(num_samples, -1)
        progress = self.fc8(progress)
        progress = torch.sigmoid(progress)
        progress = progress.reshape(B, S)

        return progress


class ProgressNetFeatures(nn.Module):
    def __init__(self, embedding_size: int, p_dropout: float) -> None:
        super(ProgressNetFeatures, self).__init__()
        self.fc7 = nn.Linear(embedding_size, 64)
        self.fc7_dropout = nn.Dropout(p=p_dropout)
        self.lstm1 = nn.LSTM(64, 64, 1, batch_first=True)
        self.lstm2 = nn.LSTM(64, 32, 1, batch_first=True)
        self.fc8 = nn.Linear(32, 1)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        B, S, _ = x.shape
        num_samples = B * S

        x = x.reshape(num_samples, -1)
        x = self.fc7(x)
        x = self.fc7_dropout(x)
        x = torch.relu(x)

        x = x.reshape(B, S, -1)
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x = x.reshape(num_samples, -1)
        x = torch.relu(x)

        x = self.fc8(x)
        x = torch.sigmoid(x)
        x = x.reshape(B, S)

        return x


class ProgressNetBoundingBoxes(nn.Module):
    def __init__(self, embedding_size: int, p_dropout: float, device: torch.device) -> None:
        super(ProgressNetBoundingBoxes, self).__init__()
        self.device = device

        self.spp = SpatialPyramidPooling([1, 2, 3, 4])
        self.spp_fc = nn.Linear(90, embedding_size)
        self.spp_dropout = nn.Dropout(p=p_dropout)

        self.roi_fc = nn.Linear(3*7*7, embedding_size)
        self.roi_dropout = nn.Dropout(p=p_dropout)

        self.fc7 = nn.Linear(embedding_size*2, 64)
        self.fc7_dropout = nn.Dropout(p=p_dropout)

        self.lstm1 = nn.LSTM(64, 64, 1, batch_first=True)
        self.lstm2 = nn.LSTM(64, 32, 1, batch_first=True)

        self.fc8 = nn.Linear(32, 1)

    def forward(self, frames: torch.FloatTensor, boxes: torch.FloatTensor) -> torch.FloatTensor:
        B, S, C, H, W = frames.shape
        num_samples = B * S

        flat_frames = frames.reshape(num_samples, C, H, W)
        flat_boxes = boxes.reshape(num_samples, 4)
        box_indices = torch.arange(start=0, end=num_samples).reshape(
            num_samples, 1).to(self.device)

        boxes_with_indices = torch.cat((box_indices, flat_boxes), dim=-1)

        pooled = self.spp(flat_frames)
        pooled = self.spp_fc(pooled)
        pooled = self.spp_dropout(pooled)
        pooled = torch.relu(pooled)

        roi = roi_pool(flat_frames, boxes_with_indices, 7)
        roi = roi.reshape(num_samples, -1)
        roi = self.roi_fc(roi)
        roi = self.roi_dropout(roi)
        roi = torch.relu(roi)

        concatenated = torch.cat((pooled, roi), dim=-1)
        concatenated = self.fc7(concatenated)
        concatenated = self.fc7_dropout(concatenated)
        concatenated = torch.relu(concatenated)

        rnn = concatenated.reshape(B, S, -1)
        rnn, _ = self.lstm1(rnn)
        rnn, _ = self.lstm2(rnn)
        rnn = torch.relu(rnn)

        progress = rnn.reshape(num_samples, -1)
        progress = self.fc8(progress)
        progress = torch.sigmoid(progress)
        progress = progress.reshape(B, S)

        return progress

class ProgressNetCategories(nn.Module):
    def __init__(self, embedding_size: int, num_categories: int, p_dropout: float) -> None:
        super(ProgressNetCategories, self).__init__()
        self.categories_fc = nn.Linear(num_categories, embedding_size)
        self.categories_dropout = nn.Dropout(p=p_dropout)

        self.fc7 = nn.Linear(embedding_size*2, 64)
        self.fc7_dropout = nn.Dropout(p=p_dropout)
        self.lstm1 = nn.LSTM(64, 64, 1, batch_first=True)
        self.lstm2 = nn.LSTM(64, 32, 1, batch_first=True)
        self.fc8 = nn.Linear(32, 1)

    def forward(self, data: torch.FloatTensor, categories: torch.FloatTensor) -> torch.FloatTensor:
        B, S, _ = data.shape
        num_samples = B * S

        categories = categories.reshape(num_samples, -1)
        categories = self.categories_fc(categories)
        categories = self.categories_dropout(categories)
        categories = torch.relu(categories)

        data = data.reshape(num_samples, -1)

        x = torch.cat((data, categories), dim=-1)
        x = self.fc7(x)
        x = self.fc7_dropout(x)
        x = torch.relu(x)

        x = x.reshape(B, S, -1)
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x = x.reshape(num_samples, -1)
        x = torch.relu(x)

        x = self.fc8(x)
        x = torch.sigmoid(x)
        x = x.reshape(B, S)

        return x

class ProgressNetFeatures2D(nn.Module):
    def __init__(self, embedding_size: int, p_dropout: float) -> None:
        super(ProgressNetFeatures2D, self).__init__()
        self.fc7 = nn.Linear(embedding_size, 64)
        self.fc7_dropout = nn.Dropout(p=p_dropout)
        self.lstm1 = nn.Linear(64, 64)
        self.lstm2 = nn.Linear(64, 32)
        self.fc8 = nn.Linear(32, 1)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        B, S, _ = x.shape
        num_samples = B * S

        x = x.reshape(num_samples, -1)
        x = self.fc7(x)
        x = self.fc7_dropout(x)
        x = torch.relu(x)


        x = self.lstm1(x)
        x = self.lstm2(x)

        x = torch.relu(x)

        x = self.fc8(x)
        x = torch.sigmoid(x)
        x = x.reshape(B, S)

        return x

class ProgressNetBoundingBoxes2D(nn.Module):
    def __init__(self, embedding_size: int, p_dropout: float, device: torch.device) -> None:
        super(ProgressNetBoundingBoxes2D, self).__init__()
        self.device = device

        self.spp = SpatialPyramidPooling([1, 2, 3, 4])
        self.spp_fc = nn.Linear(90, embedding_size)
        self.spp_dropout = nn.Dropout(p=p_dropout)

        self.roi_fc = nn.Linear(3*7*7, embedding_size)
        self.roi_dropout = nn.Dropout(p=p_dropout)

        self.fc7 = nn.Linear(embedding_size*2, 64)
        self.fc7_dropout = nn.Dropout(p=p_dropout)

        self.lstm1 = nn.Linear(64, 64)
        self.lstm2 = nn.Linear(64, 32)

        self.fc8 = nn.Linear(32, 1)

    def forward(self, frames: torch.FloatTensor, boxes: torch.FloatTensor) -> torch.FloatTensor:
        B, S, C, H, W = frames.shape
        num_samples = B * S

        flat_frames = frames.reshape(num_samples, C, H, W)
        flat_boxes = boxes.reshape(num_samples, 4)
        box_indices = torch.arange(start=0, end=num_samples).reshape(
            num_samples, 1).to(self.device)

        boxes_with_indices = torch.cat((box_indices, flat_boxes), dim=-1)

        pooled = self.spp(flat_frames)
        pooled = self.spp_fc(pooled)
        pooled = self.spp_dropout(pooled)
        pooled = torch.relu(pooled)

        roi = roi_pool(flat_frames, boxes_with_indices, 7)
        roi = roi.reshape(num_samples, -1)
        roi = self.roi_fc(roi)
        roi = self.roi_dropout(roi)
        roi = torch.relu(roi)

        concatenated = torch.cat((pooled, roi), dim=-1)
        concatenated = self.fc7(concatenated)
        concatenated = self.fc7_dropout(concatenated)
        concatenated = torch.relu(concatenated)

        rnn = self.lstm1(rnn)
        rnn = self.lstm2(rnn)
        rnn = torch.relu(rnn)

        progress = self.fc8(progress)
        progress = torch.sigmoid(progress)
        progress = progress.reshape(B, S)

        return progress

class ProgressResNet(nn.Module):
    def __init__(self) -> None:
        super(ProgressResNet, self).__init__()
        self.resnet = models.resnet18()
        self.resnet.fc = nn.Linear(512, 64)

        self.lstm1 = nn.LSTM(64, 64, 1, batch_first=True)
        self.lstm2 = nn.LSTM(64, 32, 1, batch_first=True)

        self.fc8 = nn.Linear(32, 1)

    def forward(self, frames: torch.FloatTensor, *args, **kwargs) -> torch.FloatTensor:
        B, S, C, H, W = frames.shape
        num_samples = B * S

        flat_frames = frames.reshape(num_samples, C, H, W)

        embedded = self.resnet(flat_frames)

        rnn = embedded.reshape(B, S, -1)
        rnn, _ = self.lstm1(rnn)
        rnn, _ = self.lstm2(rnn)
        rnn = torch.relu(rnn)

        progress = rnn.reshape(num_samples, -1)
        progress = self.fc8(progress)
        progress = torch.sigmoid(progress)
        progress = progress.reshape(B, S)

        return progress

class ProgressResNetIndices(nn.Module):
    def __init__(self, device: torch.device) -> None:
        self.device = device
        super(ProgressResNetIndices, self).__init__()
        self.resnet = models.resnet18()
        self.resnet.fc = nn.Linear(512, 64)

        self.lstm1 = nn.LSTM(64, 64, 1, batch_first=True)
        self.lstm2 = nn.LSTM(64, 32, 1, batch_first=True)

        self.fc8 = nn.Linear(32+1, 1)

    def forward(self, frames: torch.FloatTensor, *args, **kwargs) -> torch.FloatTensor:
        B, S, C, H, W = frames.shape
        num_samples = B * S

        flat_frames = frames.reshape(num_samples, C, H, W)

        embedded = self.resnet(flat_frames)

        rnn = embedded.reshape(B, S, -1)
        rnn, _ = self.lstm1(rnn)
        rnn, _ = self.lstm2(rnn)
        rnn = torch.relu(rnn)

        indices = torch.arange(1, S+1, 1, requires_grad=True, dtype=torch.float32, device=self.device).reshape(num_samples, -1)
        progress = rnn.reshape(num_samples, -1)
        progress = torch.cat((progress, indices), dim=-1)
        progress = self.fc8(progress)
        progress = torch.sigmoid(progress)
        progress = progress.reshape(B, S)

        return progress

def vgg(cfg, i, batch_norm=False):
    layers = []
    in_channels = i
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'C':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    conv6 = nn.Conv2d(512, 512, kernel_size=3, padding=6, dilation=6)
    conv7 = nn.Conv2d(512, 1024, kernel_size=1)
    layers += [pool5, conv6,
               nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]
    return layers

base = {
    '300': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 256, 256, 256, 'M',
            512, 512, 512],
    '512': [],
}

class ProgressNetBoundingBoxesVGG(nn.Module):
    def __init__(self, embedding_size: int, p_dropout: float, device: torch.device) -> None:
        super(ProgressNetBoundingBoxesVGG, self).__init__()
        self.device = device

        vgg_layers = tuple(vgg(base[str(300)], 3))
        self.vgg = nn.Sequential(*vgg_layers)
        self.vgg.load_state_dict(torch.load('/home/frans/Datasets/ucf24/train_data/vgg16_reducedfc.pth'))
        for param in self.vgg.parameters():
            param.requires_grad = False
        # self.vgg = nn.Sequential(
        #     nn.Conv2d(3, 32, 3, 2), nn.ReLU(),
        #     nn.Conv2d(32, 64, 3, 2), nn.ReLU(),
        #     nn.MaxPool2d(2, 2),
        #     nn.Conv2d(64, 128, 2, 2), nn.ReLU(),
        #     nn.Conv2d(128, 128, 2, 2), nn.ReLU(),
        # )
        # self.vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)

        self.spp = SpatialPyramidPooling([1, 2, 3])
        self.spp_fc = nn.Linear(14336, embedding_size)
        self.spp_dropout = nn.Dropout(p=p_dropout)

        self.roi_fc = nn.Linear(16384, embedding_size)
        self.roi_dropout = nn.Dropout(p=p_dropout)

        self.fc7 = nn.Linear(embedding_size*2, 64)
        self.fc7_dropout = nn.Dropout(p=p_dropout)

        self.lstm1 = nn.LSTM(64, 64, 1, batch_first=True)
        self.lstm2 = nn.LSTM(64, 32, 1, batch_first=True)

        self.fc8 = nn.Linear(32, 1)

    def forward(self, frames: torch.FloatTensor, boxes: torch.FloatTensor) -> torch.FloatTensor:
        B, S, C, H, W = frames.shape
        num_samples = B * S

        frames = frames.reshape(num_samples, C, H, W)
        boxes = boxes.reshape(num_samples, 4)
        box_indices = torch.arange(start=0, end=num_samples).reshape(
            num_samples, 1).to(self.device)

        boxes = torch.cat((box_indices, boxes), dim=-1)
        del box_indices
        frames = self.vgg(frames)

        roi = roi_pool(frames, boxes, 4)
        del boxes

        frames = self.spp(frames)

        frames = self.spp_fc(frames)
        frames = self.spp_dropout(frames)
        frames = torch.relu(frames)

        roi = roi.reshape(num_samples, -1)
        roi = self.roi_fc(roi)
        roi = self.roi_dropout(roi)
        roi = torch.relu(roi)

        roi = torch.cat((frames, roi), dim=-1)
        del frames

        roi = self.fc7(roi)
        roi = self.fc7_dropout(roi)
        roi = torch.relu(roi)

        roi = roi.reshape(B, S, -1)
        roi, _ = self.lstm1(roi)
        roi, _ = self.lstm2(roi)
        roi = torch.relu(roi)

        roi = roi.reshape(num_samples, -1)
        roi = self.fc8(roi)
        roi = torch.sigmoid(roi)
        roi = roi.reshape(B, S)

        return roi