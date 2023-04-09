import random


class Removal:
    def __init__(self, p: float = 0.5, p_removal=0.05):
        self.p = p
        self.p_removal = p_removal

    def __call__(self, names):
        if random.random() > self.p:
            return names

        new_names = []
        for name in names:
            if random.random() > self.p_removal:
                new_names.append(name)

        if len(new_names) == 0:
            return names
        return new_names
