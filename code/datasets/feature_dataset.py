from torch.utils.data import Dataset
import os
import torch
from typing import List
from tqdm import tqdm

from .utils import load_splitfile

class FeatureDataset(Dataset):
    def __init__(self, root: str, data_type: str, splitfile: str, flat: bool = False, transform = None) -> None:
        super().__init__()
        self.transform = transform
        split_path = os.path.join(root, 'splitfiles', splitfile)
        splitnames = load_splitfile(split_path)
        self.flat = flat
        self.data = self._get_data(os.path.join(root, data_type), splitnames, flat)

    @staticmethod
    def _get_data(root: str, splitnames: List[str], flat: bool) -> List[str]:
        data = []
        for video_name in splitnames:
            path = os.path.join(root, f'{video_name}.txt')
            with open(path) as f:
                video_data = f.readlines()

            video_data = torch.FloatTensor([list(map(float, row.split(' '))) for row in video_data])
            S, _ = video_data.shape
            progress = torch.arange(0, S) / S
            if flat:
                for i, (embedding, p) in enumerate(zip(video_data, progress)):
                    data.append((f'{video_name}_{i}', embedding, p))
            else:
                data.append((video_name, video_data, progress))
        return data


    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]