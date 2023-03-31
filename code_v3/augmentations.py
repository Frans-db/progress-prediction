import random

class Subsection:
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, names: list[str]) -> list[str]:
        if random.random() > self.p:
            return names

        start = random.randint(0, len(names) - 2)
        end = random.randint(start+1, len(names) - 1)

        return names[start:end]

class Subsample:
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, names: list[str]) -> list[str]:
        if random.random() > self.p:
            return names

        fps = random.randint(1, 10)
        return names[::fps]

class Removal:
    def __init__(self, p: float = 0.5, p_removal=0.05):
        self.p = p
        self.p_removal = p_removal

    def __call__(self, names: list[str]) -> list[str]:
        if random.random() > self.p:
            return names

        new_names = []
        for name in names:
            if random.random() > self.p_removal:
                new_names.append(name)

        if len(new_names) == 0:
            return names
        return new_names