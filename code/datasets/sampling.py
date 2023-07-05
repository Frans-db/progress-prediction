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
    
class Truncate:
    def __init__(self, max_length: int) -> None:
        self.max_length = max_length

    def __call__(self, indices: List[int]) -> List[int]:
        if len(indices) < self.max_length:
            return indices
        return indices[:self.max_length]
    
class Middle:
    def __call__(self, indices: List[int]) -> List[int]:
        num_frames = len(indices)
        middle_index = num_frames // 2
        return [indices[middle_index]]