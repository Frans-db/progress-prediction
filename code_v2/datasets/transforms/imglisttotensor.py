import torch
import torchvision.transforms as transforms
from tqdm import tqdm

class ImglistToTensor(torch.nn.Module):
    def __init__(self, dim: int = 1, transform = None) -> None:
        super().__init__()
        self.dim = dim
        if transform is None:
            self.transform = transforms.ToTensor()
        else:
            self.transform = transform

    def forward(self, img_list):
        return torch.stack([self.transform(pic) for pic in img_list], dim=self.dim)