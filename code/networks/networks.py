import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import roi_pool, roi_align
import torchvision.models as models
import os

from .pyramid_pooling import SpatialPyramidPooling

cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M', 512, 512, 512]

def resnet_forward_impl(self, x):
    # See note [TorchScript super()]
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)

    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)

    return x

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
        self.device = device
        self.debug = args.debug
        self.channels = args.backbone_channels
        self.depth = args.backbone_depth + 1
        self.roi_type = args.roi_type
        self.roi_size = args.roi_size
        self.roi_scale = args.roi_scale

        # create vgg net
        if args.backbone == 'vgg512':
            self.vgg = models.vgg16()
        elif args.backbone == 'vgg1024':
            vgg_layers = vgg(cfg, 3)
            self.vgg = nn.ModuleList(vgg_layers)
        # load vgg weights
        if args.backbone_name:
            model_path = os.path.join(args.data_root, args.dataset, 'train_data', args.backbone_name)
            self.vgg.load_state_dict(torch.load(model_path))
        # extract vgg features from pytorch vgg
        if hasattr(self.vgg, 'features'):
            self.vgg = nn.ModuleList(self.vgg.features)
        # freeze vgg weights
        if not args.backbone_gradients:
            for param in self.vgg.parameters():
                param.requires_grad = False

        # spp
        num_pools = sum(map(lambda x: x**2, args.pooling_layers))
        self.spp = SpatialPyramidPooling(args.pooling_layers)
        self.spp_fc = nn.Linear(self.channels * num_pools, args.embedding_size)
        self.spp_dropout = nn.Dropout(p=args.dropout_chance)
        # roi
        self.roi_fc = nn.Linear(self.channels * (self.roi_size**2), args.embedding_size)
        self.roi_dropout = nn.Dropout(p=args.dropout_chance)
        # progressnet
        self.fc7 = nn.Linear(2*args.embedding_size, 64)
        self.fc7_dropout = nn.Dropout(p=args.dropout_chance)
        self.lstm1 = nn.LSTM(64, 64, 1, batch_first=True)
        self.lstm2 = nn.LSTM(64, 32, 1, batch_first=True)
        # freeze all weights (except fc8) when finetuning)
        if args.finetune:
            for param in self.parameters():
                param.requires_grad = False

        self.fc8 = nn.Linear(32, 1)

    def forward(self, frames, boxes):
        B, S, C, H, W = frames.shape
        num_samples = B * S
        # reshaping frames & adding indices to boxes
        frames = frames.reshape(num_samples, C, H, W)
        if self.debug: print('Frames', frames.shape)
        boxes = boxes.reshape(num_samples, 4)
        box_indices = torch.arange(start=0, end=num_samples, device=self.device).reshape(num_samples, 1)
        boxes = torch.cat((box_indices, boxes), dim=-1)
        # vgg untill depth
        for i in range(self.depth):
            frames = self.vgg[i](frames)
        if self.debug: print(f'Frames (vgg to depth {self.depth})', frames.shape)
        # spp
        pooled = self.spp(frames)
        if self.debug: print(f'Pooled', pooled.shape)
        pooled = torch.relu(self.spp_fc(pooled))
        pooled = self.spp_dropout(pooled)
        # roi
        if self.roi_type == 'pool':
            roi = roi_pool(frames, boxes, self.roi_size, self.roi_scale)
        elif self.roi_type == 'align':
            roi = roi_align(frames, boxes, self.roi_size, self.roi_scale)
        if self.debug: print(f'ROI (type {self.roi_type} / size {self.roi_size})', roi.shape)

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