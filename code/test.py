import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import roi_pool
import torchvision.models as models
import os

path_old = '/home/frans/Datasets/ucf24/train_data/vgg_old.pth'
path_new = '/home/frans/Datasets/ucf24/train_data/vgg_512_features.pth'

old = torch.load(path_old)
new = torch.load(path_new)

old_model = models.vgg16().features
old_model.load_state_dict(old)

new_model = models.vgg16()
new_model.load_state_dict(new)
new_model = new_model.features


print(old_model.state_dict() == new_model.state_dict())

