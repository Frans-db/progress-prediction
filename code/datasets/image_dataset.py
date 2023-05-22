from torch.utils.data import Dataset
import os
import torch
from typing import List
from PIL import Image
import numpy as np
import random

from .utils import load_splitfile


class ImageDataset(Dataset):
    def __init__(
        self,
        root: str,
        data_dir: str,
        splitfile: str,
        flat: bool,

        indices: bool,
        indices_normalizer: int,
        shuffle: bool,

        transform=None,
        sample_transform=None,
    ) -> None:
        super().__init__()
        self.transform = transform
        self.sample_transform = sample_transform
        self.splitfile = splitfile
        self.shuffle = shuffle
        split_path = os.path.join(root, "splitfiles", splitfile)
        splitnames = sorted(load_splitfile(split_path))
        self.flat = flat
        self.indices = indices
        self.indices_normalizer = indices_normalizer
        self.data, self.lengths = self._get_data(os.path.join(root, data_dir), splitnames, flat)

    def _get_data(self, root: str, splitnames: List[str], flat: bool) -> List[str]:
        data = []
        lengths = []
        for video_name in splitnames:
            video_path = os.path.join(root, video_name)
            frame_names = sorted(os.listdir(video_path))
            frame_paths = [os.path.join(video_path, frame_name) for frame_name in frame_names]
            num_frames = len(frame_paths)
            progress = torch.arange(1, num_frames + 1) / num_frames

            lengths.append(num_frames)

            if flat:
                for i, (path, p) in enumerate(zip(frame_paths, progress)):
                    data.append((f"{video_name}_{i}", path, p))
            else:
                if self.shuffle:
                    random.shuffle(frame_paths)
                data.append((video_name, frame_paths, progress))

        return data, lengths

    def print_statistics(self):
        for video_name, frames, _ in self:
            print(frames.shape)
            print(torch.sum(frames, (0, 2, 3)))


    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index):
        video_name, frame_paths, progress = self.data[index]
        if self.flat:
            frame = Image.open(frame_paths)
            if self.transform:
                frame = self.transform(frame)
            return video_name, frame, progress
        else: # TODO: Better Subsampling
            frames = []
            num_frames = len(frame_paths)

            indices = list(range(num_frames))
            if self.sample_transform:
                indices = self.sample_transform(indices)

            frame_paths = np.array(frame_paths)
            frame_paths = frame_paths[indices]
            for frame_path in frame_paths:
                frame = Image.open(frame_path)
                if self.transform:
                    frame = self.transform(frame)
                frames.append(frame)
            
            frames = torch.stack(frames)
            if self.indices:
                C, H, W = frames[0].shape
                frames = torch.arange(0, num_frames, dtype=torch.float32).reshape(num_frames, 1, 1, 1).repeat(1, C, H, W)

            frames = frames[indices]
            progress = progress[indices]

            return video_name, frames, progress
