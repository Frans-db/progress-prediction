import os
from typing import List, Tuple
import torch
from PIL import Image
from torch.utils.data import Dataset
from torch.nn.functional import one_hot
import numpy as np


from .base_dataset import BaseDataset


class ProgressFeatureDataset(BaseDataset):
    def __init__(self, root: str, data_type: str, split_file: str, transform=None) -> None:
        super(ProgressFeatureDataset, self).__init__(root, data_type, split_file, transform)
        data, progress = self._load_data(self.split_names, self.data_root)
        self.data = data
        self.progress = progress

    @property
    def lengths(self) -> List[int]:
        return [item.shape[0] for item in self.data]

    @property
    def average_length(self) -> float:
        lengths = self.lengths
        return sum(lengths) / len(lengths)

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

        if 'transform' in self.transform:
            video_data = self.transform['transform'](video_data)
        if 'data_transform' in self.transform:
            video_data = self.transform['data_transform'](video_data)
        if 'sample_transform' in self.transform:
            indices = self.transform['sample_transform'](indices)
        video_data = video_data[indices]
        video_progress = video_progress[indices]

        return video_name, video_data, video_progress


class ProgressVideoDataset(BaseDataset):
    def __init__(self, root: str, data_type: str, split_file: str, transform=None) -> None:
        super(ProgressVideoDataset, self).__init__(root, data_type, split_file, transform)
        paths, progress = self._get_frame_paths(self.split_names, self.data_root)
        self.paths = paths
        self.progress = progress

    @property
    def lengths(self) -> List[int]:
        return [len(paths) for paths in self.paths]

    @property
    def average_length(self) -> float:
        lengths = self.lengths
        return sum(lengths) / len(lengths)

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
        return len(self.paths)

    def __getitem__(self, index: int) -> Tuple[str, torch.FloatTensor, torch.FloatTensor]:
        video_name = self.split_names[index]
        frame_paths = np.array(self.paths[index])
        video_progress = self.progress[index]

        num_frames = len(frame_paths)
        indices = list(range(num_frames))

        # frame_paths = frame_paths[indices]
        # video_progress = video_progress[indices]

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

        return video_name, frames[indices], video_progress[indices]

class ProgressCategoryDataset(BaseDataset):
    def __init__(self, root: str, data_type: str, category_directory: str, num_categories: int, split_file: str, transform=None) -> None:
        super(ProgressCategoryDataset, self).__init__(root, data_type, split_file, transform)
        category_root = os.path.join(self.root, category_directory)
        data, categories, progress = self._load_data(self.split_names, self.data_root, category_root, num_categories)
        self.data = data
        self.categories = categories
        self.progress = progress

    @property
    def lengths(self) -> List[int]:
        return [item.shape[0] for item in self.data]

    @property
    def average_length(self) -> float:
        lengths = self.lengths
        return sum(lengths) / len(lengths)

    @staticmethod
    def _load_data(split_names: List[str], data_root: str, category_root: str, num_categories: int) -> Tuple[List[torch.FloatTensor], List[torch.FloatTensor], List[torch.FloatTensor]]:
        category_mapping = []
        data, video_categories, progress = [], [], []
        for filename in split_names:
            feature_path = os.path.join(data_root, f'{filename}.txt')
            category_path = os.path.join(category_root, f'{filename}.txt')
            with open(category_path, 'r') as f:
                categories = f.readlines()
                categories = [line.strip() for line in categories]
                mapped_categories = []
                for category in categories:
                    if category in category_mapping:
                        mapped_categories.append(category_mapping.index(category))
                    else:
                        mapped_categories.append(len(category_mapping))
                        category_mapping.append(category)
                mapped_categories = torch.LongTensor(mapped_categories)
                mapped_categories = one_hot(torch.Tensor(mapped_categories), num_classes=num_categories).float()

            with open(feature_path, 'r') as f:
                lines = f.readlines()
            video_data = []
            for line in lines:
                line_data = map(lambda x: float(x.strip()), line.split(' '))
                video_data.append(list(line_data))
            video_length = len(video_data)
            video_progress = [(i+1) / video_length for i in range(video_length)]

            data.append(torch.FloatTensor(video_data))
            video_categories.append(mapped_categories)
            progress.append(torch.FloatTensor(video_progress))
        return data, video_categories, progress

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Tuple[str, torch.FloatTensor, torch.FloatTensor]:
        video_name = self.split_names[index]
        video_data = self.data[index]
        video_categories = self.categories[index]
        video_progress = self.progress[index]

        indices = list(range(video_data.shape[0]))
        if 'sample_transform' in self.transform:
            indices = self.transform['sample_transform'](indices)
        video_data = video_data[indices]
        video_categories = video_categories[indices]
        video_progress = video_progress[indices]

        if 'transform' in self.transform:
            video_data = self.transform['transform'](video_data)
        if 'data_transform' in self.transform:
            video_data = self.transform['data_transform'](video_data)

        return video_name, video_data, video_categories, video_progress
