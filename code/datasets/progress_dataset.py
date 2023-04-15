import os
from typing import List, Tuple
import torch
from PIL import Image
from torch.utils.data import Dataset
import numpy as np

from .base_dataset import BaseDataset


class ProgressFeatureDataset(BaseDataset):
    def __init__(self, root: str, data_type: str, split_file: str, transform=None, sample_transform=None) -> None:
        super(ProgressFeatureDataset, self).__init__(root, data_type, split_file, transform, sample_transform)
        data, progress = self._load_data(self.split_names, self.data_root)
        self.data = data
        self.progress = progress

    @staticmethod
    def _load_data(split_names: List[str], data_root: str) -> Tuple[List[torch.FloatTensor], List[torch.FloatTensor]]:
        data, progress = [], []
        for filename in split_names:
            feature_path = os.path.join(data_root, f'{filename}.txt')
            with open(feature_path, 'r') as f:
                lines = f.readlines()
            video_data = []
            for line in lines:
                line_data = map(lambda x: float(x.strip()), line.split(' '))
                video_data.append(list(line_data))
            video_length = len(video_data)
            video_progress = [
                (i+1) / video_length for i in range(video_length)]

            data.append(torch.FloatTensor(video_data))
            progress.append(torch.FloatTensor(video_progress))
        return data, progress

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Tuple[str, torch.FloatTensor, torch.FloatTensor]:
        video_name = self.split_names[index]
        video_data = self.data[index]
        video_progress = self.progress[index]

        indices = list(range(video_data.shape[0]))
        if self.sample_transform:
            indices = self.sample_transform(indices)
        video_data = video_data[indices]
        video_progress = video_progress[indices]

        if self.transform:
            video_data = self.transform(video_data)

        return video_name, video_data, video_progress


class ProgressVideoDataset(BaseDataset):
    def __init__(self, root: str, data_type: str, split_file: str, transform=None, sample_transform=None) -> None:
        super(ProgressVideoDataset, self).__init__(root, data_type, split_file, transform, sample_transform)
        paths, progress = self._get_frame_paths(self.split_names, self.data_root)
        self.paths = paths
        self.progress = progress

    @staticmethod
    def _get_frame_paths(split_names: List[str], data_root: str) -> Tuple[List[str], List[torch.FloatTensor]]:
        paths, progress = [], []
        for filename in split_names:
            video_path = os.path.join(data_root, filename)
            frame_names = sorted(os.listdir(video_path))
            frame_paths = [os.path.join(video_path, frame_name)
                           for frame_name in frame_names]
            video_length = len(frame_paths)
            video_progress = [(i+1) / video_length for i in range(video_length)]

            paths.append(frame_paths)
            progress.append(torch.FloatTensor(video_progress))
        return paths, progress

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Tuple[str, torch.FloatTensor, torch.FloatTensor]:
        video_name = self.split_names[index]
        frame_paths = np.array(self.paths[index])
        video_progress = self.progress[index]

        num_frames = len(frame_paths)
        indices = list(range(num_frames))
        if self.sample_transform:
            indices = self.sample_transform(indices)
        frame_paths = frame_paths[indices]
        video_progress = video_progress[indices]

        frames = []
        for frame_path in frame_paths:
            frame = Image.open(frame_path).convert('RGB')
            if self.transform:
                frame = self.transform(frame)
            frames.append(frame)

        return video_name, torch.stack(frames), video_progress
