import os
import torch
from PIL import Image
import numpy as np


class ProgressDataset(torch.utils.data.Dataset):
    def __init__(self,
                 root_path: str,
                 num_segments: int,
                 frames_per_segment: int,
                 sample_every: int,
                 imagefile_template: str = 'img_{:05d}.png',
                 test_mode: bool = False,
                 transform=None) -> None:
        super().__init__()
        self.root_path = root_path
        self.items = os.listdir(self.root_path)
        self.num_segments = num_segments
        self.frames_per_segment = frames_per_segment
        self.sample_every = sample_every
        self.imagefile_template = imagefile_template
        self.transform = transform
        self.test_mode = test_mode

        self._check_samples()

    def _check_samples(self):
        """
        Check if each video has enough frames to be sampled
        """
        for video_name in self.items:
            path = os.path.join(self.root_path, video_name)
            frame_names = os.listdir(path)
            num_frames = len(frame_names)

            if num_frames <= 0:
                print(
                    f"\nDataset Warning: video {video_name} seems to have zero RGB frames on disk!\n")
            elif num_frames < self.num_segments * self.frames_per_segment:
                print(f"\nDataset Warning: video {video_name} has {num_frames} frames "
                      f"but the dataloader is set up to load "
                      f"(num_segments={self.num_segments})*(frames_per_segment={self.frames_per_segment})"
                      f"={self.num_segments * self.frames_per_segment} frames. Dataloader will throw an "
                      f"error when trying to load this video.\n")

    def _get_start_indices(self, num_frames: int) -> np.ndarray:
        """
        For each segment, choose a start index from where frames
        are to be loaded from.

        Args:
            num_frames: Number of frames the video has
        Returns:
            List of indices of where the frames of each
            segment are to be loaded from.
        """
        # choose start indices that are perfectly evenly spread across the video frames.
        if self.test_mode:
            distance_between_indices = (
                num_frames - self.frames_per_segment + 1) / float(self.num_segments)

            start_indices = np.array([int(distance_between_indices / 2.0 + distance_between_indices * x)
                                      for x in range(self.num_segments)])
        # randomly sample start indices that are approximately evenly spread across the video frames.
        else:
            max_valid_start_index = (
                num_frames - self.frames_per_segment + 1) // self.num_segments
            start_indices = np.multiply(list(range(self.num_segments)), max_valid_start_index) + \
                np.random.randint(max_valid_start_index,
                                  size=self.num_segments)

        return start_indices

    def __getitem__(self, idx):
        item = self.items[idx]
        item_directory = os.path.join(self.root_path, item)
        image_files = sorted(os.listdir(item_directory))
        num_frames = len(image_files)

        start_indices = self._get_start_indices(num_frames)

        images = []
        labels = []
        for start_index in start_indices:
            indices = list(range(start_index, start_index+self.frames_per_segment))[::self.sample_every]
            for i in indices:
                image_name = self.imagefile_template.format(i)
                image_path = os.path.join(item_directory, image_name)
                images.append(self._load_image(image_path))
                labels.append((i + 1) / num_frames)

        if self.transform is not None:
            images = self.transform(images)
        return images, np.array(labels)

    def _load_image(self, path: str) -> Image.Image:
        return Image.open(path).convert('RGB')

    def __len__(self):
        return len(self.items)
