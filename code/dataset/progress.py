import os
import torch
from PIL import Image

class ProgressDataset(torch.utils.data.Dataset):
    def __init__(self, 
                 root_path: str,
                 num_segments: int = 3,
                 frames_per_segment: int = 1,
                 transform = None) -> None:
        super().__init__()
        self.root_path = root_path
        self.items = os.listdir(self.root_path)
        
        self.num_segments = num_segments
        self.frames_per_segment = frames_per_segment

        self.transform = transform

        self._check_samples()

    def _check_samples(self):
        for video_directory in self.items:
            path = os.path.join(self.root_path, video_directory)
            frame_names = os.listdir(path)
            num_frames = len(frame_names)
            
    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        item_directory = os.path.join(self.root_path, item)
        image_files = sorted(os.listdir(item_directory))
        num_images = len(image_files)
        

        images = []
        labels = []
        for i,filename in enumerate(image_files):
            image_path = os.path.join(item_directory, filename)
            images.append(Image.open(image_path))
            labels.append((i+1) / num_images)

        if self.transform is not None:
            return self.transform(images), labels
        return images, labels
            
    def _load_image(self, path: str) -> Image.Image:
        return Image.open(path).convert('RGB')