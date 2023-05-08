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

        rsd_type: str,
        fps: float,

        transform=None,
        sample_transform = None
    ) -> None:
        super().__init__()
        self.splitfile = splitfile
        self.flat = flat
        self.indices = indices
        self.indices_normalizer = indices_normalizer
        self.rsd_type = rsd_type
        self.fps = fps
        self.transform = transform
        self.sample_transform = sample_transform

        split_path = os.path.join(root, "splitfiles", splitfile)
        self.splitnames = load_splitfile(split_path)

        self.data, self.lengths = self._get_data(os.path.join(root, data_dir))

    def _get_data(self, root: str) -> List[str]:
        data = []
        lengths = []
        for video_name in self.splitnames:
            path = os.path.join(root, f"{video_name}.txt")
            with open(path) as f:
                video_data = f.readlines()

            video_data = torch.FloatTensor(
                [list(map(float, row.split(" "))) for row in video_data]
            )

            S, F = video_data.shape
            lengths.append(S)

            if self.indices:
                indices = torch.arange(0, S, dtype=torch.float32).reshape(S, 1).repeat(1, F) / self.indices_normalizer
                video_data = torch.cat((video_data, indices), dim=-1)

            progress = torch.arange(1, S + 1) / S

            video_length = (S / self.fps)
            if self.rsd_type == 'minutes':
                video_length = video_length / 60
            rsd = progress * video_length
            rsd = torch.flip(rsd, dims=(0, ))

            indices = list(range(S))
            if self.sample_transform:
                indices = self.sample_transform(indices)
            video_data = video_data[indices]
            progress = progress[indices]
            rsd = rsd[indices]

            if self.flat:
                for i, (embedding, p, rsd_val) in enumerate(zip(video_data, progress, rsd)):
                    data.append((f"{video_name}_{i}", embedding, p, rsd_val))
            else:
                data.append((video_name, video_data, progress, rsd))
        return data, lengths

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index):
        name, data, progress, rsd = self.data[index]
        if self.rsd_type != "none":
            return name, data, torch.flip(rsd, dims=(0, )), rsd, progress
        return name, data, progress
