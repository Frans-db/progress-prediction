from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import os

class VideoFrameDataset(Dataset):
    def __init__(self, root_dir: str, num_videos: int, offset: int = 0, transform = None) -> None:
        super().__init__()
        self.root_dir = root_dir
        self.num_videos = num_videos
        self.offset = offset
        self.transform = transform

        self.frames_paths = []
        self.frames_progress = []

        for i in range(num_videos):
            video_path = f'{self.root_dir}/{(i+self.offset):05d}'
            frame_names = sorted(os.listdir(video_path))
            num_frames = len(frame_names)

            frame_paths = [f'{video_path}/{frame_name}' for frame_name in frame_names]
            frame_progress = [(frame_index+1) / num_frames for frame_index in range(num_frames)]

            self.frames_paths += frame_paths
            self.frames_progress += frame_progress


    def __len__(self):
        return len(self.frames_paths)

    def __getitem__(self, index):
        video_index = index // 90
        frame_index = index % 90

        image = Image.open(self.frames_paths[index]).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, self.frames_progress[index]