import os
from typing import List, Tuple, Dict
import torch
from PIL import Image
from torch.utils.data import Dataset
import pickle
import numpy as np

from .base_dataset import BaseDataset


class BoundingBoxDataset(BaseDataset):
    def __init__(self, root: str, data_type: str, split_file: str, annotation_file: str, transform=None, image_size: Tuple[int, int] = (320, 240)) -> None:
        super(BoundingBoxDataset, self).__init__(root, data_type, split_file, transform)
        self.image_size = image_size
        annotation_path = os.path.join(self.root, 'splitfiles', annotation_file)
        tube_names, frame_paths, boxes, progress = self._load_tubes(self.split_names, annotation_path, self.data_root)
        self.tube_names = tube_names
        self.frame_paths = frame_paths
        self.boxes = boxes
        self.progress = progress
        self.num_features = 2

    @property
    def lengths(self) -> List[int]:
        return [len(paths) for paths in self.frame_paths]

    @property
    def average_length(self) -> float:
        lengths = self.lengths
        return sum(lengths) / len(lengths)

    def __len__(self) -> int:
        return len(self.tube_names)

    def __getitem__(self, index: int) -> Tuple[str, torch.FloatTensor, torch.FloatTensor]:
        tube_name = self.tube_names[index]
        frame_paths = np.array(self.frame_paths[index])
        boxes = self.boxes[index]
        progress = self.progress[index]

        num_frames = len(frame_paths)
        indices = list(range(num_frames))

        frames = []
        for frame_path in frame_paths:
            frame = Image.open(frame_path).convert('RGB')
            if 'transform' in self.transform:
                frame = self.transform['transform'](frame)
            frames.append(frame)
        frames = torch.stack(frames)
        if 'data_transform' in self.transform:
            frames = self.transform['data_transform'](frames)
        if 'sample_transform' in self.transform:
            indices = self.transform['sample_transform'](indices)

        return tube_name, frames[indices], boxes[indices], progress[indices]

    def _load_tubes(self, split_names: List[str], annotation_path: str, data_root: str) -> Tuple[List[str], List[List[str]], List[torch.FloatTensor], List[torch.FloatTensor]]:
        with open(annotation_path, 'rb') as f:
            database = pickle.load(f)

        tube_names, frame_paths, boxes, progress = [], [], [], []
        for video_name in database:
            if video_name not in split_names:
                continue
            video_path = os.path.join(data_root, video_name)
            tubes = database[video_name]['annotations']
            for tube in tubes:
                tube_frame_paths, tube_boxes = self._load_tube(tube, video_path)
                num_frames = len(tube_frame_paths)
                tube_progress = [(i+1) / num_frames for i in range(num_frames)]

                tube_names.append(f'{video_name}_{tube["sf"]}_{tube["ef"]}')
                frame_paths.append(tube_frame_paths)
                boxes.append(tube_boxes)
                progress.append(torch.FloatTensor(tube_progress))

        return tube_names, frame_paths, boxes, progress

    def _load_tube(self, tube: Dict, video_path: str) -> Tuple[List[str], torch.FloatTensor]:
        frame_paths = []
        for frame_id in range(tube['sf'], tube['ef']):
            frame_name = f'{(frame_id+1):05d}.jpg'
            frame_path = os.path.join(video_path, frame_name)
            frame_paths.append(frame_path)

        boxes = tube['boxes'].astype('float32')
        boxes = self._convert_boxes(boxes)

        return frame_paths, boxes

    def _convert_boxes(self, boxes: np.ndarray) -> torch.FloatTensor:
        # convert boxes to (x_min, y_min, x_max, y_max)
        boxes[:, 2] += boxes[:, 0]
        boxes[:, 3] += boxes[:, 1]

        return torch.FloatTensor(boxes)