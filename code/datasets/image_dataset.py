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
        flat: bool = False,

        subsample_fps: int = 1,

        random_data: bool = False,
        indices: bool = False,
        indices_normalizer: int = 1,
        shuffle: bool = False,

        transform=None,
        sample_transform=None,
    ) -> None:
        super().__init__()
        self.transform = transform
        self.sample_transform = sample_transform
        self.subsample_fps = subsample_fps
        self.random = random_data
        self.splitfile = splitfile
        self.shuffle = shuffle
        split_path = os.path.join(root, "splitfiles", splitfile)
        splitnames = (load_splitfile(split_path))
        self.flat = flat
        self.indices = indices
        self.index_to_index = []
        self.indices_normalizer = indices_normalizer
        self.data, self.lengths = self._get_data(os.path.join(root, data_dir), splitnames, flat)

    def _get_data(self, root: str, splitnames: List[str], flat: bool) -> List[str]:
        data = []
        lengths = []
        for video_name in splitnames:
            video_path = os.path.join(root, video_name)
            frame_names = sorted(os.listdir(video_path))
            frame_paths = [os.path.join(video_path, frame_name) for frame_name in frame_names]
            # video subsampling
            frame_paths = frame_paths[::self.subsample_fps]
            num_frames = len(frame_paths)
            progress = torch.arange(1, num_frames + 1) / num_frames

            lengths.append(num_frames)

            if flat:
                for i, (path, p) in enumerate(zip(frame_paths, progress)):
                    data.append((f"{video_name}_{i}", path, p))
                    self.index_to_index.append(i)
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
            if self.indices:
                frame_index = self.index_to_index[index]
                frame = torch.full_like(frame, frame_index) / self.indices_normalizer
            if self.random:
                frame = torch.rand_like(frame)
            return video_name, frame, progress
        else: # TODO: Better Subsampling
            frames = []
            num_frames = len(frame_paths)

            indices = list(range(num_frames))
            if self.sample_transform:
                indices = self.sample_transform(indices)

            frame_paths = np.array(frame_paths)
            frame_paths = frame_paths[indices]
            for index, path in zip(indices, frame_paths):
                frame = Image.open(path)
                if self.transform:
                    frame = self.transform(frame)
                if self.indices:
                    frame = torch.full_like(frame, index) / self.indices_normalizer
                frames.append(frame)
                # frame.close()
            
            if self.transform:
                frames = torch.stack(frames)
            if self.random:
                frames = torch.rand_like(frames)

            progress = progress[indices]

            return video_name, frames, progress
