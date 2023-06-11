from typing import List
import random

class Subsection:
    def __call__(self, indices: List[int]) -> List[int]:
        length = len(indices)
        if length == 1:
            return indices
        start = random.randrange(0, length - 1)
        remaining = length - start
        duration = random.randint(1, remaining)

        return indices[start:start+duration]

class Subsample:
    def __call__(self, indices: List[int]) -> List[int]:
        fps = random.randint(1, 10)
        return indices[::fps]