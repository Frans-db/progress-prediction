import torch
import torchvision.transforms as transforms
from PIL import Image
from typing import List

class SwapDimensions(torch.nn.Module):
    def __call__(self, img_list):
        shape = img_list.shape
        assert len(shape) == 4
        return img_list.reshape(shape[1], shape[0], shape[2], shape[3])