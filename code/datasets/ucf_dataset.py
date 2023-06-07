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
        bounding_boxes: bool,
        flat: bool,
        subsample_fps: int,
        random: int,
        indices: bool,
        indices_normalizer: int,
        rsd_type: str,
        fps: float,
        transform=None,
    ) -> None:
        super().__init__()

        self.bounding_boxes = bounding_boxes
        self.flat = flat
        self.subsample_fps = subsample_fps
        self.random = random
        self.indices = indices
        self.indices_normalizer = indices_normalizer
        self.rsd_type = rsd_type
        self.fps = fps
        self.transform = transform
        self.splitfile = splitfile
        self.index_to_index = []

        split_path = os.path.join(root, "splitfiles", splitfile)
        self.splitnames = load_splitfile(split_path)

        data_root = os.path.join(root, data_dir)
        database_path = os.path.join(root, "splitfiles/pyannot.pkl")
        self.data, self.lengths = self._load_database(data_root, database_path)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int):
        name, paths, boxes, progress, rsd = self.data[index]
        if self.flat:
            frame = Image.open(paths)
            if self.transform:
                frame = self.transform(frame)
            if self.indices:
                frame_index = self.index_to_index[index]
                frame = torch.full_like(frame, frame_index) / self.indices_normalizer
            if self.bounding_boxes and self.rsd_type != 'none':
                return name, frame, boxes, rsd, progress
            elif self.bounding_boxes:
                return name, frame, boxes, progress
            elif self.rsd_type != 'none':
                return name, frame, rsd, progress
            else:
                return name, frame, progress
        else:
            frames = []
            for path in paths:
                frame = Image.open(path)
                if self.transform:
                    frame = self.transform(frame)
                frames.append(frame)

            frames = torch.stack(frames)
            if self.bounding_boxes and self.rsd_type != 'none':
                return name, frames, boxes, rsd, progress
            elif self.bounding_boxes:
                return name, frames, boxes, progress
            elif self.rsd_type != 'none':
                return name, frames, rsd, progress
            else:
                return name, frames, progress

    def _load_database(self, data_root: str, database_path: str):
        with open(database_path, "rb") as f:
            database = pickle.load(f)

        data = []
        lengths = []
        for video_name in database:
            if video_name not in self.splitnames:
                continue
            for tube_index, tube in enumerate(database[video_name]["annotations"]):
                boxes = tube["boxes"]
                boxes[:, 2] += boxes[:, 0]
                boxes[:, 3] += boxes[:, 1]
                boxes = torch.FloatTensor(boxes.astype(float))

                boxes[:, 0] *= (224 / 320)
                boxes[:, 2] *= (224 / 240)
                boxes[:, 1] *= (224 / 320)
                boxes[:, 3] *= (224 / 240)

                paths = []
                for frame_index in range(tube["sf"], tube["ef"]):
                    frame_path = os.path.join(
                        data_root, video_name, f"{frame_index+1:05d}.jpg"
                    )
                    paths.append(frame_path)

                # Video subsampling
                boxes = boxes[::self.subsample_fps, :]
                paths = paths[::self.subsample_fps]

                num_frames = boxes.shape[0]
                lengths.append(num_frames)
                progress = torch.arange(1, num_frames + 1) / num_frames

                video_length = (num_frames / self.fps)
                if self.rsd_type == 'minutes':
                    video_length = video_length / 60
                rsd = progress * video_length
                rsd = torch.flip(rsd, dims=(0, ))

                if self.flat:
                    for i, (frame_path, box, p, rsd_val) in enumerate(
                        zip(paths, boxes, progress, rsd)
                    ):
                        data.append(
                            (f"{video_name}_{tube_index}_{i}", frame_path, box, p, rsd_val)
                        )
                        self.index_to_index.append(i)
                else:
                    data.append((f"{video_name}_{tube_index}", paths, boxes, progress, rsd))

        return data, lengths
