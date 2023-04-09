import random


class Subsection:
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, names):
        if random.random() > self.p:
            return names

        start = random.randint(0, len(names) - 2)
        end = random.randint(start+1, len(names) - 1)

        return names[start:end]
