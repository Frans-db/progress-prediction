from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import os

class Toy3DDataset(Dataset):
    def __init__(self, root_dir: str, num_videos: int, offset: int = 0, transform=None) -> None:
        super().__init__()
        self.root_dir = root_dir
        self.num_videos = num_videos
        self.offset = offset
        self.transform = transform


    def __len__(self):
        return self.num_videos

    def __getitem__(self, index):
        new_index = index + self.offset
        images = []
        labels = []

        video_path = f'{self.root_dir}/{new_index:05d}'
        frame_names = sorted(os.listdir(video_path))
        num_frames = len(frame_names)

        for frame_index,frame_name in enumerate(frame_names):
            path = f'{video_path}/{frame_name}'
            images.append(Image.open(path).convert('RGB'))
            labels.append((frame_index + 1) / num_frames)

        if self.transform:
            images = self.transform(images)

        return images, np.array(labels)