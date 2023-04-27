import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import roi_pool, roi_align
import torchvision.models as models
import os

from .pyramid_pooling import SpatialPyramidPooling

cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M', 512, 512, 512]

def init_weights(m: nn.Module) -> None:
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.uniform_(m.bias, a=0, b=0)
    elif isinstance(m, nn.LSTM):
        nn.init.xavier_uniform_(m.weight_hh_l0)
        nn.init.xavier_uniform_(m.weight_ih_l0)
        nn.init.uniform_(m.bias_ih_l0, a=0, b=0)
        nn.init.uniform_(m.bias_hh_l0, a=0, b=0)

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


class RSDFlat(nn.Module):
    def __init__(self, args, device) -> None:
        super(RSDFlat, self).__init__()
        self.device = device
        self.channels = args.backbone_channels

        # create resnet
        if args.backbone == 'resnet18':
            self.resnet = models.resnet18()
        elif args.backbone == 'resnet34':
            self.resnet = models.resnet34()
        elif args.backbone == 'resnet50':
            self.resnet = models.resnet50()
        elif args.backbone == 'resnet101':
            self.resnet = models.resnet101()
        elif args.backbone == 'resnet152':
            self.resnet = models.resnet152()
        # load resnet weights
        if args.backbone_name:
            model_path = os.path.join(args.data_root, args.dataset, 'train_data', args.backbone_name)
            self.resnet.load_state_dict(torch.load(model_path))

        self.resnet.fc = nn.Linear(self.channels, 1)


    def forward(self, frames, *args, **kwargs):
        B, C, H, W = frames.shape

        progress = torch.sigmoid(self.resnet(frames))
    
        return progress.reshape(B)

class RSDNet(nn.Module):
    def __init__(self, args, device) -> None:
        super(RSDNet, self).__init__()
        self.device = device
        self.resnet_dropout = nn.Dropout(p=args.dropout_chance)
        self.lstm_dropout = nn.Dropout(p=args.dropout_chance)
        self.lstm = nn.LSTM(args.backbone_channels, args.embedding_size, 1, batch_first=True)

        self.progress_head = nn.Linear(args.embedding_size+1, 1)
        self.rsd_head = nn.Linear(args.embedding_size+1, 1)

    def forward(self, features, elapsed, *args, **kwargs):
        B, S, F = features.shape

        features = self.resnet_dropout(features)
        features, _ = self.lstm(features)
        features = self.lstm_dropout(features)

        elapsed = elapsed.reshape(B, S, 1)
        features = torch.cat((features, elapsed), dim=-1)
        features = features.reshape(B*S, -1)

        rsd = self.rsd_head(features)
        progress = self.progress_head(features)

        return rsd.reshape(B, S), progress.reshape(B, S)
        