def load_splitfile(split_path: str):
    split_names = []
    with open(split_path, 'r') as f:
        for line in f.readlines():
            video_name = line.strip()
            split_names.append(video_name)
    return split_names