from torch.utils.data import Dataset
import os
import torch
from typing import List
from PIL import Image

from .utils import load_splitfile


class ImageDataset(Dataset):
    def __init__(
        self,
        root: str,
        data_dir: str,
        splitfile: str,
        flat: bool,
        indices: bool,
        transform=None,
    ) -> None:
        super().__init__()
        self.transform = transform
        self.splitfile = splitfile
        split_path = os.path.join(root, "splitfiles", splitfile)
        splitnames = load_splitfile(split_path)
        self.flat = flat
        self.data, self.lengths = self._get_data(os.path.join(root, data_dir), splitnames, flat)

    @staticmethod
    def _get_data(root: str, splitnames: List[str], flat: bool) -> List[str]:
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
                data.append((video_name, frame_paths, progress))

        return data, lengths

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index):
        video_name, frame_paths, progress = self.data[index]
        if self.flat:
            frame = Image.open(frame_paths)
            if self.transform:
                frame = self.transform(frame)
            return video_name, frame, progress
        else: # TODO: Subsampling
            frames = []
            for frame_path in frame_paths:
                frame = Image.open(frame_path)
                if self.transform:
                    frame = self.transform(frame)
                frames.append(frame)
            
            return video_name, torch.stack(frames), progress
