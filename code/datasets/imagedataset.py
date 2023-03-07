import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import pickle
import os
from os.path import join
from PIL import Image

from .utils import load_splitfile


class ImageDataset(Dataset):
    def __init__(self, data_root: str, data_type: str, splitfile_path: str, transform=None):
        super(ImageDataset, self).__init__()
        # create paths
        self.data_root = join(data_root, data_type)
        self.splitfile_path = join(data_root, splitfile_path)
        # load split
        split_names = load_splitfile(self.splitfile_path)
        self.frame_paths, self.progress = self._get_data(split_names)
        self.transform = transform

    # Dunder Methods

    def __getitem__(self, index):
        frame_path = self.frame_paths[index]
        frame = Image.open(frame_path)

        if self.transform:
            frame = self.transform(frame)

        return frame, torch.FloatTensor(self.progress[index])

    def __len__(self) -> int:
        return len(self.frame_paths)

    # Helper Methods

    def _get_data(self, split_names):
        frame_paths = []
        progress = []
        for video_name in split_names:
            video_dir = join(self.data_root, video_name)

            frame_names = sorted(os.listdir(video_dir))
            num_frames = len(frame_names)
            for frame_index, frame_name in enumerate(frame_names):
                frame_paths.append(join(video_dir, frame_name))
                progress.append((frame_index + 1) / num_frames)

        return frame_paths, progress

def main():
    dataset = ImageDataset(
        '/home/frans/Datasets/ucf24',
        'rgb-images',
        'splitfiles/testlist01.txt',
        transform=transforms.ToTensor()
    )
    frame, progress = dataset[0]
    print(frame)
    print(frame.shape)
    print(progress)


if __name__ == '__main__':
    main()
