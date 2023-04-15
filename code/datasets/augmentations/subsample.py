import random
from typing import List


class Subsample:
    def __init__(self, p: float = 0.5) -> None:
        self.p = p

    def __call__(self, indices: List[int]) -> List[int]:
        if random.random() > self.p:
            return indices

        fps = random.randint(1, 10)
        return indices[::fps]
