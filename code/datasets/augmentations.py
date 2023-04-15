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


class Ones:
    def __call__(self, data: torch.FloatTensor) -> torch.FloatTensor:
        return torch.ones_like(data, dtype=torch.float32, requires_grad=True)


class Randoms:
    def __call__(self, data: torch.FloatTensor) -> torch.FloatTensor:
        return torch.rand_like(data, dtype=torch.float32, requires_grad=True)


class Subsample:
    def __init__(self, p: float = 0.5) -> None:
        self.p = p

    def __call__(self, indices: List[int]) -> List[int]:
        if random.random() > self.p:
            return indices

        fps = random.randint(1, 10)
        return indices[::fps]


class Subsection:
    def __init__(self, p: float = 0.5) -> None:
        self.p = p

    def __call__(self, indices: List[int]) -> List[int]:
        if random.random() > self.p:
            return indices

        start = random.randint(0, len(indices) - 2)
        end = random.randint(start+1, len(indices) - 1)

        return indices[start:end]
