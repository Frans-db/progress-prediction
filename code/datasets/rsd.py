import torch

from .progress import ProgressDataset
from .transforms import ImglistToTensor

class RSDDataset(ProgressDataset):
    def __init__(self, data_root, data_type, splitfile_path, fps: int = 25, mode='seconds', transform=None):
        super(RSDDataset, self).__init__(data_root, data_type, splitfile_path, transform=transform)
        self.fps = fps

    # Dunder Methods

    def __getitem__(self, index):
        video_name, frames, progress_values = super().__getitem__(index)

        total_time = len(frames) / self.fps
        rsd_values = [total_time - p * total_time for p in progress_values]

        return video_name, frames, torch.FloatTensor(rsd_values), progress_values

    # Data Analysis Methods

    def get_max_video_length(self):
        length = self.get_max_video_frame_length() / self.fps
        if self.mode == 'minutes':
            length /= 60
        return length

    def get_average_video_length(self):
        length = self.get_average_video_frame_length() / self.fps
        if self.mode == 'minutes':
            length /= 60
        return length

def main():
    dataset = RSDDataset(
        '/mnt/hdd/datasets/ucf24',
        'rgb-images',
        'splitfiles/testlist01.txt',
        transform=ImglistToTensor(dim=0)
    )
    video_name, frames, rsd_values, progress_values = dataset[0]
    print(video_name)
    print(frames.shape)
    print(rsd_values)
    print(progress_values)

if __name__ == '__main__':
    main()