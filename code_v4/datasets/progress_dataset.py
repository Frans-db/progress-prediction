import os
import torch
from torch.utils.data import Dataset
from typing import Tuple, List
from tqdm import tqdm

class ProgressDataset(Dataset):
    def __init__(self, root: str, data_type: str, split_file: str, sample_augmentations = None) -> None:
        super(ProgressDataset, self).__init__()
        self.root = root
        self.data_root = os.path.join(self.root, data_type)
        self.files = self._load_split_file(os.path.join(self.root, split_file))
        self.data = self._load_files(self.files)

        self.sample_augmentations = sample_augmentations

    @property
    def embedding_size(self) -> int:
        return self.data[0].shape[-1]

    @property
    def lengths(self) -> List[int]:
        lengths = []
        for sample in self.data:
            lengths.append(sample.shape[0])
        return lengths

    @property
    def average_length(self) -> float:
        lengths = self.lengths
        return sum(lengths) / len(lengths)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[str, torch.FloatTensor, torch.FloatTensor]:
        data = self.data[idx]
        progress = torch.arange(1, data.shape[0] + 1) / data.shape[0]

        indices = list(range(data.shape[0]))
        if self.sample_augmentations:
            indices = self.sample_augmentations(indices)

        return self.files[idx], data[indices], progress[indices]

    def _load_split_file(self, path: str) -> List[str]:
        with open(path, 'r') as f:
            data = f.readlines()
        return [row.strip() for row in data]

    def _load_files(self, files: List[str]) -> List[torch.FloatTensor]:
        data = []
        for filename in tqdm(files):
            path = f'{os.path.join(self.data_root, filename)}'
            if not path.endswith('.txt'):
                path = path + '.txt'
            with open(path, 'r') as f:
                lines = f.readlines()
                video_data = []
                for line in lines:
                    line_data = list(
                        map(lambda x: float(x.strip()), line.split(' ')))
                    video_data.append(line_data)
                data.append(torch.FloatTensor(video_data))
        return data

    def get_action_labels(self, video_name: str) -> List[int]:
        with open(os.path.join(self.root, 'labels', video_name), 'r') as f:
            labels = [int(line) for line in f.readlines()]
        return labels
