import random
from typing import List
import torch

class Randoms:
    def __call__(self, data: torch.FloatTensor) -> torch.FloatTensor:
        return torch.rand_like(data, dtype=torch.float32, requires_grad=True)
