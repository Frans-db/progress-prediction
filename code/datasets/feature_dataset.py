import os
from typing import List, Tuple, Dict
import torch
from PIL import Image
from torch.utils.data import Dataset
import pickle
import numpy as np


class FeatureDataset(Dataset):
    def __init__(self, root: str, data_type: str, split_file: str, transform=None) -> None:
        super(FeatureDataset, self).__init__()
        self.transform = transform
        self.split_names = self._load_splitfile(os.path.join(root, 'splitfiles', split_file))
        self.data = self._load_data(os.path.join(root, data_type), self.split_names)
        
    def _load_splitfile(self, path: str) -> List[str]:
        with open(path) as f:
            names = f.readlines()
            names = [name.strip() for name in names]
        return names

    def _load_data(self, root: str, split_names: List[str]):
        all_data = []
        for name in split_names:
            path = os.path.join(root, f'{name}.txt')
            video_data = []
            with open(path) as f:
                data = f.readlines()
            for line in data:
                line = list(map(float, line.split(' ')))
                line = torch.FloatTensor(line)
                video_data.append(line)
            video_data = torch.stack(video_data)
            S = video_data.shape[0]
            progress = torch.arange(1, S+1) / S
            rsd = torch.arange(S-1, -1, -1) / S * (S / 60)

            all_data.append((video_data, rsd, progress))
        return all_data


    def __len__(self) -> int:
        return len(self.split_names)

    def __getitem__(self, index):
        data, rsd, progress = self.data[index]
        
        return data, rsd, progress
