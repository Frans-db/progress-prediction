import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import roi_pool
import torchvision.models as models
import os

from .pyramid_pooling import SpatialPyramidPooling

cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M', 512, 512, 512]

# This function is derived from torchvision VGG make_layers()
# https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
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
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    layers += [pool5, conv6,
               nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]
    return tuple(layers)


class ProgressNet(nn.Module):
    def __init__(self, args, device) -> None:
        super(ProgressNet, self).__init__()
        channels = 512
        self.device = device
        # create vgg net
        if args.basemodel == 'vgg512':
            self.vgg = models.vgg16()
            if 'old' in args.basemodel_name:
                self.vgg = self.vgg.features
        elif args.basemodel == 'vgg1024':
            channels = channels * 2
            vgg_layers = vgg(cfg, 3)
            self.vgg = nn.Sequential(*vgg_layers)
        # load vgg weights
        if args.basemodel_name:
            model_path = os.path.join(args.data_root, args.dataset, 'train_data', args.basemodel_name)
            self.vgg.load_state_dict(torch.load(model_path))
        # extract vgg features from pytorch vgg
        if hasattr(self.vgg, 'features'):
            self.vgg = self.vgg.features
        # freeze vgg weights
        if not args.basemodel_gradients:
            for param in self.vgg.parameters():
                param.requires_grad = False

        # spp
        num_pools = sum(map(lambda x: x**2, args.pooling_layers))
        self.spp = SpatialPyramidPooling(args.pooling_layers)
        self.spp_fc = nn.Linear(channels * num_pools, args.embedding_size)
        self.spp_dropout = nn.Dropout(p=args.dropout_chance)
        # roi
        self.roi_size = args.roi_size
        self.roi_fc = nn.Linear(channels * (self.roi_size**2), args.embedding_size)
        self.roi_dropout = nn.Dropout(p=args.dropout_chance)
        # progressnet
        self.fc7 = nn.Linear(2*args.embedding_size, 64)
        self.fc7_dropout = nn.Dropout(p=args.dropout_chance)

        self.lstm1 = nn.LSTM(64, 64, 1, batch_first=True)
        self.lstm2 = nn.LSTM(64, 32, 1, batch_first=True)

        self.fc8 = nn.Linear(32, 1)

    def forward(self, frames, boxes):
        B, S, C, H, W = frames.shape
        num_samples = B * S
        # reshaping frames & adding indices to boxes
        frames = frames.reshape(num_samples, C, H, W)
        boxes = boxes.reshape(num_samples, 4)
        box_indices = torch.arange(start=0, end=num_samples, device=self.device).reshape(num_samples, 1)
        boxes = torch.cat((box_indices, boxes), dim=-1)
        # vgg
        frames = self.vgg(frames)
        # spp
        pooled = self.spp(frames)
        pooled = torch.relu(self.spp_fc(pooled))
        pooled = self.spp_dropout(pooled)
        # roi
        roi = roi_pool(frames, boxes, self.roi_size, 0.03)
        roi = roi.reshape(num_samples, -1)
        roi = torch.relu(self.roi_fc(roi))
        roi = self.roi_dropout(roi)
        # concatenating
        concatenated = torch.cat((pooled, roi), dim=-1)
        # progressnet
        concatenated = torch.relu(self.fc7(concatenated))
        concatenated = self.fc7_dropout(concatenated)

        concatenated = concatenated.reshape(B, S, -1)
        concatenated, _ = self.lstm1(concatenated)
        concatenated, _ = self.lstm2(concatenated)
        concatenated = concatenated.reshape(num_samples, -1)
        
        progress = torch.sigmoid(self.fc8(concatenated))
        return progress.reshape(B, S)

class ProgressNetPooling(nn.Module):
    def __init__(self, args, device) -> None:
        super(ProgressNetPooling, self).__init__()
        self.device = device

        # spp
        num_pools = sum(map(lambda x: x**2, args.pooling_layers))
        self.spp = SpatialPyramidPooling(args.pooling_layers)
        self.spp_fc = nn.Linear(3 * num_pools, args.embedding_size)
        self.spp_dropout = nn.Dropout(p=args.dropout_chance)
        # roi
        self.roi_size = args.roi_size
        self.roi_fc = nn.Linear(3 * (self.roi_size**2), args.embedding_size)
        self.roi_dropout = nn.Dropout(p=args.dropout_chance)
        # progressnet
        self.fc7 = nn.Linear(2*args.embedding_size, 64)
        self.fc7_dropout = nn.Dropout(p=args.dropout_chance)

        self.lstm1 = nn.LSTM(64, 64, 1, batch_first=True)
        self.lstm2 = nn.LSTM(64, 32, 1, batch_first=True)

        self.fc8 = nn.Linear(32, 1)

    def forward(self, frames, boxes):
        B, S, C, H, W = frames.shape
        num_samples = B * S
        # reshaping frames & adding indices to boxes
        frames = frames.reshape(num_samples, C, H, W)
        boxes = boxes.reshape(num_samples, 4)
        box_indices = torch.arange(start=0, end=num_samples, device=self.device).reshape(num_samples, 1)
        boxes = torch.cat((box_indices, boxes), dim=-1)
        # vgg
        # frames = self.vgg(frames)
        # spp
        pooled = self.spp(frames)
        pooled = torch.relu(self.spp_fc(pooled))
        pooled = self.spp_dropout(pooled)
        # roi
        roi = roi_pool(frames, boxes, self.roi_size)
        roi = roi.reshape(num_samples, -1)
        roi = torch.relu(self.roi_fc(roi))
        roi = self.roi_dropout(roi)
        # concatenating
        concatenated = torch.cat((pooled, roi), dim=-1)
        # progressnet
        concatenated = torch.relu(self.fc7(concatenated))
        concatenated = self.fc7_dropout(concatenated)

        concatenated = concatenated.reshape(B, S, -1)
        concatenated, _ = self.lstm1(concatenated)
        concatenated, _ = self.lstm2(concatenated)
        concatenated = concatenated.reshape(num_samples, -1)
        
        progress = torch.sigmoid(self.fc8(concatenated))
        return progress.reshape(B, S)

# class Conv(nn.Module): # pytorch vgg16 features model & roi
#     def __init__(self, args, device) -> None:
#         super(Conv, self).__init__()
#         self.device = device
#         self.vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
#         for param in self.vgg.parameters():
#             param.requires_grad = False

#         self.roi_size = args.roi_size
#         self.roi_fc = nn.Linear(512 * (self.roi_size**2), 1)

#     def forward(self, frames, boxes):
#         B, S, C, H, W = frames.shape
#         num_samples = B * S

#         frames = frames.reshape(num_samples, C, H, W)
#         boxes = boxes.reshape(num_samples, 4)
#         box_indices = torch.arange(start=0, end=num_samples, device=self.device).reshape(num_samples, 1)
#         boxes = torch.cat((box_indices, boxes), dim=-1)

#         shape_before = frames.shape
#         frames = self.vgg.features(frames)
#         shape_after = frames.shape

#         progress = roi_pool(frames, boxes, self.roi_size, 0.03)
#         progress = progress.reshape(num_samples, -1)
#         progress = torch.sigmoid(self.roi_fc(progress))
#         return progress.reshape(B, S)

# class Conv(nn.Module): # pytorch vgg16 features model & spp
#     def __init__(self, args) -> None:
#         super(Conv, self).__init__()
#         self.vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
#         for param in self.vgg.parameters():
#             param.requires_grad = False

        # num_pools = sum(map(lambda x: x**2, args.pooling_layers))
        # self.spp = SpatialPyramidPooling(args.pooling_layers)
        # self.spp_fc = nn.Linear(512 * num_pools, 1)

#     def forward(self, x, *args, **kwargs):
#         B, S, C, H, W = x.shape
#         num_samples = B * S
#         x = x.reshape(num_samples, C, H, W)

#         x = self.vgg.features(x)
        # x = self.spp(x)
        # x = torch.sigmoid(self.spp_fc(x))
#         return x.reshape(B, S)

# class Conv(nn.Module): # pytorch vgg16 model
#     def __init__(self, args) -> None:
#         super(Conv, self).__init__()
#         self.vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
#         for param in self.vgg.parameters():
#             param.requires_grad = False
#         self.fc = nn.Linear(1000, 1)

#     def forward(self, x, *args, **kwargs):
#         B, S, C, H, W = x.shape
#         num_samples = B * S
#         x = x.reshape(num_samples, C, H, W)

#         x = self.vgg(x)
#         x = torch.sigmoid(self.fc(x))
#         return x.reshape(B, S)

# class Conv(nn.Module): # convolutional model
#     def __init__(self, args) -> None:
#         super(Conv, self).__init__()
#         self.conv1 = nn.Conv2d(3, 6, 5)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(6, 16, 5)
#         self.fc1 = nn.Linear(16 * 5 * 7, 120)
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, 1)

#     def forward(self, x, *args, **kwargs):
#         B, S, C, H, W = x.shape
#         num_samples = B * S
#         x = x.reshape(num_samples, C, H, W)

#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = torch.flatten(x, 1) # flatten all dimensions except batch
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = torch.sigmoid(self.fc3(x))
#         return x.reshape(B, S)