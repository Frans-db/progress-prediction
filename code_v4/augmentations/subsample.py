import random


class Subsample:
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, names):
        if random.random() > self.p:
            return names

        fps = random.randint(1, 10)
        return names[::fps]
