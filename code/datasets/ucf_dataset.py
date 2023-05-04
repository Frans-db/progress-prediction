from torch.utils.data import Dataset
import os
import torch
from typing import List
from PIL import Image
import pickle
import numpy as np

from .utils import load_splitfile


class UCFDataset(Dataset):
    def __init__(
        self,
        root: str,
        data_dir: str,
        splitfile: str,
        flat: bool = False,
        transform=None,
    ) -> None:
        super().__init__()

        self.transform = transform
        split_path = os.path.join(root, "splitfiles", splitfile)
        splitnames = load_splitfile(split_path)

        self.flat = flat
        data_root = os.path.join(root, data_dir)
        database_path = os.path.join(root, "splitfiles/pyannot.pkl")
        self.data = self._load_database(data_root, database_path, splitnames, flat)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int):
        name, paths, boxes, progress = self.data[index]
        paths = np.array(paths)
        indices = list(range(len(paths)))
        if self.sample_transform:
            indices = self.sample_transform(indices)

        paths, boxes, progress = paths[indices], boxes[indices], progress[indices]
        frames = []
        for path in paths:
            frame = Image.open(path)
            if self.transform:
                frame = self.transform(frame)
            frames.append(frame)

        return name, torch.stack(frames), boxes, progress

    @staticmethod
    def _load_database(
        data_root: str, database_path: str, splitnames: List[str], flat: bool
    ):
        with open(database_path, "rb") as f:
            database = pickle.load(f)

        data = []
        for video_name in database:
            if video_name not in splitnames:
                continue
            for tube_index, tube in enumerate(database[video_name]["annotations"]):
                boxes = tube["boxes"]
                boxes[:, 2] += boxes[:, 0]
                boxes[:, 3] += boxes[:, 1]
                boxes = torch.FloatTensor(boxes.astype(float))

                boxes[:, 0] *= 224 / 320
                boxes[:, 2] *= 224 / 240
                boxes[:, 1] *= 224 / 320
                boxes[:, 3] *= 224 / 240

                num_frames = boxes.shape[0]
                paths = []
                for frame_index in range(tube["sf"], tube["ef"]):
                    frame_path = os.path.join(
                        data_root, video_name, f"{frame_index+1:05d}.jpg"
                    )
                    paths.append(frame_path)
                progress = torch.arange(1, num_frames + 1) / num_frames

                if flat:
                    for i, (frame_path, box, p) in enumerate(
                        zip(paths, boxes, progress)
                    ):
                        data.append((f"{video_name}_{tube_index}_{i}", frame_path, box, p))
                else:
                    data.append((f"{video_name}_{tube_index}", paths, boxes, progress))

        return data
