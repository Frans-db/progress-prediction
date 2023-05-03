def load_splitfile(path: str):
    with open(path) as f:
        names = f.readlines()
    return [name.strip() for name in names]
