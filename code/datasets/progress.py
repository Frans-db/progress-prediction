import torch
from torch.utils.data import Dataset
import pickle
import os
from os.path import join
from PIL import Image

from transforms import ImglistToTensor
from utils import load_splitfile


class ProgressDataset(Dataset):
    def __init__(self, data_root: str, data_type: str, splitfile_path: str, transform=None):
        super(ProgressDataset, self).__init__()
        # create paths
        self.data_root = join(data_root, data_type)
        self.splitfile_path = join(data_root, splitfile_path)
        # load split
        self.split_names = load_splitfile(self.splitfile_path)
        self.transform = transform

    # Dunder Methods

    def __getitem__(self, index):
        video_path = join(self.data_root, self.split_names[index])
        frames = self._load_frames(video_path)
        video_name = self.split_names[index]
        num_frames = len(frames)
        progress_values = [(i+1) / num_frames for i in range(num_frames)]

        if self.transform:
            frames = self.transform(frames)

        return video_name, frames, torch.FloatTensor(progress_values)

    def __len__(self) -> int:
        return len(self.split_names)

    # Helper Methods

    def _load_frames(self, video_path: str):
        frames = []
        frame_names = sorted(os.listdir(video_path))
        for frame_name in frame_names:
            frame_path = join(video_path, frame_name)
            frames.append(Image.open(frame_path))
        return frames

    # Data Analysis Methods
    # TODO: Turn some of these into properties?

    def get_video_frame_lengths(self):
        lengths = []
        for video_name in self.split_names:
            video_path = join(self.data_root, video_name)
            frames = self._load_frames(video_path)
            lengths.append(len(frames))
        return lengths

    def get_average_video_frame_length(self):
        lengths = self.get_video_frame_lengths()
        return sum(lengths) / len(lengths)

    def get_max_video_frame_length(self):
        lengths = self.get_video_frame_lengths()
        return max(lengths)


def main():
    dataset = ProgressDataset(
        '/mnt/hdd/datasets/ucf24',
        'rgb-images',
        'splitfiles/testlist01.txt',
        transform=ImglistToTensor(dim=0)
    )
    video_name, frames, progress_values = dataset[0]
    print(video_name)
    print(frames.shape)
    print(progress_values)
    print(dataset.get_average_video_frame_length(), dataset.get_max_video_frame_length())



if __name__ == '__main__':
    main()
