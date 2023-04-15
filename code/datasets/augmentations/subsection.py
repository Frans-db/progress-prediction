import random
from typing import List


class Subsection:
    def __init__(self, p: float = 0.5) -> None:
        self.p = p

    def __call__(self, indices: List[int]) -> List[int]:
        if random.random() > self.p:
            return indices

        start = random.randint(0, len(indices) - 2)
        end = random.randint(start+1, len(indices) - 1)

        return indices[start:end]
