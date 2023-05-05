from torch.utils.data import Dataset
import os
import torch
from typing import List

from .utils import load_splitfile


class FeatureDataset(Dataset):
    def __init__(
        self,
        root: str,
        data_dir: str,
        splitfile: str,
        flat: bool,
        indices: bool,
        indices_normalizer: int,
        transform=None,
    ) -> None:
        super().__init__()
        self.transform = transform
        self.splitfile = splitfile
        split_path = os.path.join(root, "splitfiles", splitfile)
        splitnames = load_splitfile(split_path)
        self.flat = flat
        self.data, self.lengths = self._get_data(os.path.join(root, data_dir), splitnames, flat, indices, indices_normalizer)

    @staticmethod
    def _get_data(root: str, splitnames: List[str], flat: bool, indices: bool, indices_normalizer: int) -> List[str]:
        data = []
        lengths = []
        for video_name in splitnames:
            path = os.path.join(root, f"{video_name}.txt")
            with open(path) as f:
                video_data = f.readlines()

            video_data = torch.FloatTensor(
                [list(map(float, row.split(" "))) for row in video_data]
            )
            S, F = video_data.shape
            lengths.append(S)
            if indices:
                video_data = torch.arange(0, S, dtype=torch.float32).reshape(S, 1).repeat(1, F) / indices_normalizer
            progress = torch.arange(1, S + 1) / S
            if flat:
                for i, (embedding, p) in enumerate(zip(video_data, progress)):
                    data.append((f"{video_name}_{i}", embedding, p))
            else:
                data.append((video_name, video_data, progress))
        return data, lengths

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]
