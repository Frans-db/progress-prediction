from torch.utils.data import Dataset
import os
import torch
from typing import List
from tqdm import tqdm

from .utils import load_splitfile


class FeatureDataset(Dataset):
    def __init__(
        self,
        root: str,
        data_dir: str,
        splitfile: str,
        flat: bool,

        subsample_fps: int,
        random: bool,

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
        self.random = random
        self.subsample_fps = subsample_fps
        self.indices = indices
        self.indices_normalizer = indices_normalizer
        self.rsd_type = rsd_type
        self.fps = fps
        self.transform = transform
        self.sample_transform = sample_transform

        split_path = os.path.join(root, "splitfiles", splitfile)
        self.splitnames = load_splitfile(split_path)

        self.data, self.lengths, self.means, self.stds = self._get_data(os.path.join(root, data_dir))
        self.stds += 1e-6

    def _get_data(self, root: str) -> List[str]:
        data = []
        lengths = []
        all_data = []
        for video_name in tqdm(self.splitnames):
            path = os.path.join(root, f"{video_name}.txt")
            with open(path) as f:
                video_data = f.readlines()
            # TODO: Video subsampling
            video_data = torch.FloatTensor(
                [list(map(float, row.split(" "))) for row in video_data]
            )
            video_data = video_data[::self.subsample_fps, :]
            S, F = video_data.shape
            lengths.append(S)


            if self.indices:
                video_data = torch.arange(0, S, dtype=torch.float32).reshape(S, 1).repeat(1, F) / self.indices_normalizer
            progress = torch.arange(1, S + 1) / S

            video_length = (S / self.fps)
            if self.rsd_type == 'minutes':
                video_length = video_length / 60
            rsd = progress * video_length
            rsd = torch.flip(rsd, dims=(0, ))

            all_data.append(video_data)

            if self.flat:
                for i, (embedding, p, rsd_val) in enumerate(zip(video_data, progress, rsd)):
                    data.append((f"{video_name}_{i}", embedding, p, rsd_val))
            else:
                data.append((video_name, video_data, progress, rsd))
        
        all_data = torch.cat(all_data)
        return data, lengths, all_data.mean(dim=0), all_data.std(dim=0)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index):
        name, data, progress, rsd = self.data[index]
        data = (data - self.means) / self.stds

        indices = list(range(data.shape[0]))
        if self.sample_transform:
            indices = self.sample_transform(indices)
            data = data[indices, :]
            progress = progress[indices]
            rsd = rsd[indices]
        if self.random:
            data = torch.rand_like(data)

        if self.rsd_type != "none":
            return name, data, torch.flip(rsd, dims=(0, )), rsd, progress
        return name, data, progress
