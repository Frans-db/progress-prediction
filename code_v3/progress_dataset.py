import torch
from torch.utils.data import Dataset
from PIL import Image
import os

class ProgressDataset(Dataset):
    def __init__(self, root: str, data_type: str, split_path: str, transform=None, sample_transform=None) -> None:
        self.root = root
        self.data_root = os.path.join(root, data_type)
        self.split_files = self._load_split(os.path.join(root, split_path))
        self.transform = transform
        self.sample_transform = sample_transform

    def get_action_labels(self, video_name: str) -> list[int]:
        with open(os.path.join(self.root, 'labels', video_name), 'r') as f:
            labels = [int(line) for line in f.readlines()]
        return labels

    def __getitem__(self, index: int) -> tuple[torch.FloatTensor, torch.FloatTensor]:
        video_name = self.split_files[index]
        video_dir = os.path.join(self.data_root, video_name)

        frame_names = sorted(os.listdir(video_dir))
        if self.sample_transform:
            frame_names = self.sample_transform(frame_names)
        num_frames = len(frame_names)

        frames = []
        progress = []
        for i, frame_name in enumerate(frame_names):
            frame_path = os.path.join(video_dir, frame_name)
            frame = Image.open(frame_path).convert('RGB')
            if self.transform:
                frame = self.transform(frame)
            frames.append(frame)
            progress.append((i+1) / num_frames)

        return video_name, torch.stack(frames), torch.FloatTensor(progress)

    def __len__(self) -> int:
        return len(self.split_files)

    def _load_split(self, split_path: str) -> list[str]:
        with open(split_path, 'r') as f:
            lines = f.readlines()
        return [line.strip() for line in lines]
