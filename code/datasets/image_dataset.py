import os
from typing import List, Tuple, Dict
import torch
from PIL import Image
from torch.utils.data import Dataset
import pickle
import numpy as np

from .base_dataset import BaseDataset


class ImageDataset(BaseDataset):
    def __init__(self, root: str, data_type: str, split_file: str, transform=None) -> None:
        super(ImageDataset, self).__init__(root, data_type, split_file, transform)
        names, frame_paths, progress = self._load_data(os.path.join(root, data_type), self.split_names)
        self.names = names
        self.frame_paths = frame_paths
        self.progress = torch.FloatTensor(progress)
        self.num_features = 1

    def _load_data(self, root: str, split_names: List[str]):
        names, frame_paths, progress = [], [], []
        for video_name in split_names:
            video_path = os.path.join(root, video_name)
            frame_names = sorted(os.listdir(video_path))
            num_frames = len(frame_names)
            video_progress = [(i+1) / num_frames for i in range(num_frames)]
            for i, (frame_name, progress_value) in enumerate(zip(frame_names, video_progress)):
                frame_path = os.path.join(video_path, frame_name)
                frame_paths.append(frame_path)
                progress.append(progress_value)
                names.append(f'{video_name}/{frame_name}')
        return names, frame_paths, progress

    def __len__(self) -> int:
        return len(self.frame_paths)

    def __getitem__(self, index: int) -> Tuple[str, torch.FloatTensor, torch.FloatTensor]:
        name = self.names[index]
        frame_path = self.frame_paths[index]
        progress = self.progress[index]

        frame = Image.open(frame_path)
        if 'transform' in self.transform:
            frame = self.transform['transform'](frame)
        return name, frame, progress