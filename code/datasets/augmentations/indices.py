import random
from typing import List
import torch


class Indices:
    def __init__(self, normalization_factor: float) -> None:
        super(Indices, self).__init__()
        self.normalization_factor = normalization_factor

    def __call__(self, data: torch.FloatTensor) -> torch.FloatTensor:
        print(data.shape)
        return data
        # return torch.rand_like(data, dtype=torch.float32, requires_grad=True)
