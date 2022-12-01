from torch.utils.data import Dataset
from PIL import Image
import numpy as np

class Toy3DDataset(Dataset):
    def __init__(self, root_dir: str, num_videos: int, offset: int = 0, frames_per_video: int = 90, transform=None) -> None:
        super().__init__()
        self.root_dir = root_dir
        self.num_videos = num_videos
        self.frames_per_video = frames_per_video
        self.offset = offset
        self.transform = transform


    def __len__(self):
        return self.num_videos

    def __getitem__(self, index):
        new_index = index + self.offset
        images = []
        labels = []
        for frame_index in range(self.frames_per_video):
            path = f'{self.root_dir}/{new_index:05d}/img_{frame_index:05d}.png'
            images.append(Image.open(path).convert('RGB'))
            labels.append((frame_index + 1) / self.frames_per_video)

        if self.transform:
            images = self.transform(images)

        return images, np.array(labels)