from torch.utils.data import Dataset
import torch
import os

class FileDataset(Dataset):
    def __init__(self, root: str, data_type: str, split_file: str):
        self.root = root
        self.data_root = os.path.join(self.root, data_type)
        self.files = self._load_split_file(os.path.join(self.root, 'splitfiles', split_file))
        self.data = self._load_files(self.files)

    @property
    def embedding_size(self) -> int:
        return self.data[0].shape[-1]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        progress = torch.arange(1, data.shape[0] + 1) / data.shape[0]

        return self.files[idx], data, progress


    def _load_split_file(self, path: str):
        with open(path, 'r') as f:
            data = f.readlines()
        return [row.strip() for row in data]

    def _load_files(self, files):
        data = []
        for filename in files:
            path = f'{os.path.join(self.data_root, filename)}.txt'
            with open(path, 'r') as f:
                lines = f.readlines()
                video_data = []
                for line in lines:
                    line_data = list(map(lambda x: float(x.strip()), line.split(' ')))
                    video_data.append(line_data)
                data.append(torch.FloatTensor(video_data))
        return data

    def get_action_labels(self, video_name: str):
        with open(os.path.join(self.root, 'labels', video_name), 'r') as f:
            labels = [int(line) for line in f.readlines()]
        return labels