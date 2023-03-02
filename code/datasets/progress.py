import torch
from torch.utils.data import Dataset
import pickle
import os
from os.path import join
from PIL import Image

from transforms import ImglistToTensor

def load_splitfile(split_path: str):
    split_names = []
    with open(split_path, 'r') as f:
        for line in f.readlines():
            video_name = line.strip()
            split_names.append(video_name)
    return split_names


class ProgressDataset(Dataset):
    def __init__(self, data_root: str, data_type: str, splitfile_path: str, transform = None):
        super(ProgressDataset, self).__init__()
        self.data_root = join(data_root, data_type)
        self.splitfile_path = join(data_root, splitfile_path)
        self.split_names = load_splitfile(self.splitfile_path)
        self.transform = transform

    def __getitem__(self, index):
        video_path = join(self.data_root, self.split_names[index])
        frames = self._load_frames(video_path)
        num_frames = len(frames)
        progress_values = torch.FloatTensor([(i+1) / num_frames for i in range(num_frames)])

        if self.transform:
            frames = self.transform(frames)

        return frames, progress_values

    def __len__(self) -> int:
        return len(self.split_names)

    def _load_frames(self, video_path: str):
        frames = []
        frame_names = sorted(os.listdir(video_path))
        for frame_name in frame_names:
            frame_path = join(video_path, frame_name)
            frames.append(Image.open(frame_path))
        return frames


def main():
    dataset = ProgressDataset(
        '/mnt/hdd/datasets/ucf24', 
        'rgb-images', 
        'splitfiles/testlist01.txt', 
        transform=ImglistToTensor(dim=0)
    )
    print(len(dataset))
    frames, progress_values = dataset[0]
    print(frames.shape)
    print(progress_values)


if __name__ == '__main__':
    main()
