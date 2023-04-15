import random
from typing import List
import torch

class Ones:
    def __call__(self, data: torch.FloatTensor) -> torch.FloatTensor:
        return torch.ones_like(data, dtype=torch.float32, requires_grad=True)
