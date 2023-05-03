from torch.utils.data import Dataset
import torch
from typing import List
from PIL import Image


class ChunkDataset(Dataset):
    def __init__(self, frame_paths: List[str], chunk_size: int, transform=None) -> None:
        super().__init__()
        self.transform = transform
        self.cache = {}

        self.video_paths = self._create_chunks(frame_paths, chunk_size)

    def __len__(self) -> int:
        return len(self.video_paths)

    def __getitem__(self, index) -> torch.FloatTensor:
        paths = self.video_paths[index]
        frames = []
        for path in paths:
            if path in self.cache:
                frame = self.cache[path]
            else:
                frame = Image.open(path)
                if self.transform:
                    frame = self.transform(frame)
                self.cache[path] = frame
            frames.append(frame)

        return torch.stack(frames, dim=1)

    def _create_chunks(
        self, frame_paths: List[str], chunk_size: int
    ) -> List[List[str]]:
        num_frames = len(frame_paths)

        video_paths = []  # Frames to chunks
        for i in range(-(chunk_size // 2), num_frames - (chunk_size // 2)):
            paths = []
            for j in range(i, i + chunk_size):
                j = min(max(j, 0), num_frames - 1)
                paths.append(frame_paths[j])
            video_paths.append(paths)

        return video_paths
