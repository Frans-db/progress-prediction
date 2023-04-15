import random
from typing import List
import torch


class Indices:
    def __init__(self, normalization_factor: float) -> None:
        super(Indices, self).__init__()
        self.normalization_factor = normalization_factor

    def __call__(self, data: torch.FloatTensor) -> torch.FloatTensor:
        S, C, H, W = data.shape
        x = torch.arange(1, S+1, 1, dtype=torch.float32).reshape(S, 1) / self.normalization_factor
        data = torch.ones_like(data, dtype=torch.float32).reshape(S, -1)
        data = data * x
        return data.reshape(S, C, H, W)


class Ones:
    def __call__(self, data: torch.FloatTensor) -> torch.FloatTensor:
        return torch.ones_like(data, dtype=torch.float32)


class Randoms:
    def __call__(self, data: torch.FloatTensor) -> torch.FloatTensor:
        return torch.rand_like(data, dtype=torch.float32)


class Subsample:
    def __init__(self, p: float = 0.5) -> None:
        self.p = p

    def __call__(self, indices: List[int]) -> List[int]:
        if random.random() > self.p:
            return indices

        fps = random.randint(1, 10)
        return indices[::fps]

    def __repr__(self) -> str:
        return f'<Subsample(p={self.p})>'

class Subsection:
    def __init__(self, p: float = 0.5) -> None:
        self.p = p

    def __call__(self, indices: List[int]) -> List[int]:
        if random.random() > self.p:
            return indices

        start = random.randint(0, len(indices) - 2)
        end = random.randint(start+1, len(indices) - 1)

        return indices[start:end]

    def __repr__(self) -> str:
        return f'<Subsection(p={self.p})>'
