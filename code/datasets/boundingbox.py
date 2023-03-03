import torch
from torch.utils.data import Dataset
import pickle
import os
from os.path import join
from PIL import Image

from .transforms import ImglistToTensor
from .utils import load_splitfile


class BoundingBoxDataset(Dataset):
    def __init__(self, data_root: str, data_type: str, annotation_path: str, splitfile_path: str, frame_name_format_function, transform=None, image_size=(320, 240)):
        super(BoundingBoxDataset, self).__init__()
        # create paths
        self.data_root = join(data_root, data_type)
        self.annotation_path = join(data_root, annotation_path)
        self.splitfile_path = join(data_root, splitfile_path)

        self.transform = transform
        self.frame_name_format_function = frame_name_format_function
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
        progress_values = [(i+1) / num_frames for i in range(num_frames)]

        if self.transform:
            frames = self.transform(frames)


        return video_name, frames, tube, torch.FloatTensor(progress_values)

    def __len__(self) -> int:
        return len(self.tubes)

    # Helper Methods

    def _load_frames(self, frame_paths: list[str]):
        frames = []
        for frame_path in frame_paths:
            frames.append(Image.open(frame_path))
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
            frame_name = self.frame_name_format_function(frame_id)
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
    dataset = BoundingBoxDataset(
        '/mnt/hdd/datasets/ucf24',
        'rgb-images',
        'splitfiles/pyannot.pkl',
        'splitfiles/testlist01.txt',
        frame_name_format_function=lambda x: f'{(x+1):05d}.jpg',
        transform=ImglistToTensor(dim=0)
    )

    print(len(dataset))
    print(dataset.get_average_tube_frame_length(), dataset.get_max_tube_framelength())
    video_name, frames, tube, progress_values = dataset[0]
    print(video_name)
    print(frames.shape, tube.shape)
    print(progress_values)

if __name__ == '__main__':
    main()
