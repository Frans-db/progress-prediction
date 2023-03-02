import torch
import torchvision.transforms as transforms

class ImglistToTensor(torch.nn.Module):
    def __init__(self, dim: int = 1) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, img_list):
        return torch.stack([transforms.functional.to_tensor(pic) for pic in img_list], dim=self.dim)