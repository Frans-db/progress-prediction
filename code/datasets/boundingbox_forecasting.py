import torch
from torch.utils.data import Dataset
import pickle
import os
from os.path import join
from PIL import Image
from typing import List

from .transforms import ImglistToTensor
from .utils import load_splitfile


class BoundingBoxForecastingDataset(Dataset):
    def __init__(self, data_root: str, data_type: str, annotation_path: str, splitfile_path: str, delta_t: int, transform=None, image_size=(320, 240)):
        super(BoundingBoxForecastingDataset, self).__init__()
        # create paths
        self.data_root = join(data_root, data_type)
        self.annotation_path = join(data_root, annotation_path)
        self.splitfile_path = join(data_root, splitfile_path)

        self.delta_t = delta_t

        self.transform = transform
        self.image_size = image_size
        # load split & tubes
        self.split_names = load_splitfile(self.splitfile_path)
        self.video_names, self.frame_paths, self.tubes = self._load_tubes()

    # Dunder Methods

    def __getitem__(self, index):
        frame_paths = self.frame_paths[index]
        frames = self._load_frames(frame_paths)
        video_name = self.video_names[index]
        tube = self.tubes[index]

        num_frames = len(frames)
        progress_values = torch.FloatTensor([(i+1) / num_frames for i in range(num_frames)])

        if self.transform:
            frames = self.transform(frames)
        
        future_frames = torch.zeros_like(frames)
        future_frames[:-self.delta_t, :, :, :] = frames[self.delta_t:, :, :, :]
        future_progress_values = torch.ones_like(progress_values)
        future_progress_values[:-self.delta_t] = progress_values[self.delta_t:]
        future_tube = torch.zeros_like(tube)
        future_tube[:-self.delta_t, :] = tube[self.delta_t:, :]

        return video_name, frames, tube, progress_values, future_frames, future_tube, future_progress_values

    def __len__(self) -> int:
        return len(self.tubes)

    # Helper Methods

    def _load_frames(self, frame_paths: List[str]):
        frames = []
        for frame_path in frame_paths:
            # pillow leaves images open for too long, causing an os error: too many files open
            # copying the image and closing the original fixes this
            # https://stackoverflow.com/questions/29234413/too-many-open-files-error-when-opening-and-loading-images-in-pillow
            # https://github.com/python-pillow/Pillow/issues/1144
            # https://github.com/python-pillow/Pillow/issues/1237
            temp_image = Image.open(frame_path)
            frames.append(temp_image.copy())
            temp_image.close()
        return frames

    def _load_tubes(self):
        with open(self.annotation_path, 'rb') as f:
            database = pickle.load(f)

        all_frame_paths = []
        all_boxes = []
        video_names = []
        for video_name in database:
            if video_name not in self.split_names:
                continue

            video_path = join(self.data_root, video_name)
            tubes = database[video_name]['annotations']

            for tube in tubes:
                frame_paths, boxes = self._load_tube(tube, video_path)

                all_frame_paths.append(frame_paths)
                all_boxes.append(boxes)
                video_names.append(video_name)

        return video_names, all_frame_paths, all_boxes

    def _load_tube(self, tube, video_path: str):
        frame_paths = []
        for frame_id in range(tube['sf'], tube['ef']):
            frame_name = f'{(frame_id+1):05d}.jpg'
            frame_path = join(video_path, frame_name)
            frame_paths.append(frame_path)

        boxes = tube['boxes'].astype('float32')
        # TODO: function modifies boxes inplace, not a big fan of this
        boxes = self._convert_boxes(boxes)

        return frame_paths, boxes

    def _convert_boxes(self, boxes):
        # convert boxes to (x_min, y_min, x_max, y_max)
        boxes[:, 2] += boxes[:, 0]
        boxes[:, 3] += boxes[:, 1]
        # make box sizes relative to image size
        boxes[:, 0] /= self.image_size[0]
        boxes[:, 2] /= self.image_size[0]
        boxes[:, 1] /= self.image_size[1]
        boxes[:, 3] /= self.image_size[1]

        return torch.FloatTensor(boxes)

    # Data Analysis Methods
    # TODO: Turn some of these into properties?

    def get_tube_frame_lengths(self):
        lengths = []
        for tube in self.tubes:
            lengths.append(len(tube))
        return lengths

    def get_average_tube_frame_length(self):
        lengths = self.get_tube_frame_lengths()
        return sum(lengths) / len(lengths)

    def get_max_tube_frame_length(self):
        lengths = self.get_tube_frame_lengths()
        return max(lengths)


def main():
    import matplotlib.pyplot as plt

    dataset = BoundingBoxDataset(
        '/mnt/hdd/datasets/ucf24',
        'rgb-images',
        'splitfiles/pyannot.pkl',
        'splitfiles/testlist01.txt',
        transform=ImglistToTensor(dim=0)
    )

    lengths = dataset.get_tube_frame_lengths()
    plot_data = {}
    for length in lengths:
        if length not in plot_data:
            plot_data[length] = 0
        plot_data[length] += 1

    # plt.ba
    

if __name__ == '__main__':
    main()
