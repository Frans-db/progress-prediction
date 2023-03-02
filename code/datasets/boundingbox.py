import torch
from torch.utils.data import Dataset
import pickle
import os
from os.path import join
from PIL import Image

from transforms import ImglistToTensor
from utils import load_splitfile


class BoundingBoxDataset(Dataset):
    def __init__(self, data_root: str, data_type: str, annotation_path: str, splitfile_path: str, transform=None, frame_name_format: str = 'img_{0:05d}.png', image_size=(320, 240)):
        super(BoundingBoxDataset, self).__init__()
        # create paths
        self.data_root = join(data_root, data_type)
        self.annotation_path = join(data_root, annotation_path)
        self.splitfile_path = join(data_root, splitfile_path)

        self.transform = transform
        self.frame_name_format = frame_name_format
        self.image_size = image_size
        # load split & tubes
        self.split_names = load_splitfile(self.splitfile_path)
        self.frame_paths, self.tubes = self._load_tubes()

    # Dunder Methods

    def __getitem__(self, index):
        video_path = join(self.data_root, self.split_names[index])
        frames = self._load_frames(video_path)
        num_frames = len(frames)
        progress_values = [(i+1) / num_frames for i in range(num_frames)]

        if self.transform:
            frames = self.transform(frames)

        return frames, torch.FloatTensor(progress_values)

    def __len__(self) -> int:
        return len(self.tubes)

    # Helper Methods

    def _load_tubes(self):
        with open(self.annotation_path, 'rb') as f:
            database = pickle.load(f)

        all_frame_paths = []
        all_boxes = []
        for video_name in database:
            if video_name not in self.split_names:
                continue

            video_path = join(self.data_root, video_name)
            tubes = database[video_name]['annotations']

            for tube in tubes:
                frame_paths, boxes = self._load_tube(tube, video_path)

                all_frame_paths.append(frame_paths)
                all_boxes.append(torch.FloatTensor(boxes))

        return all_frame_paths, all_boxes

    def _load_tube(self, tube, video_path: str):
        frame_paths = []
        for frame_id in range(tube['sf'], tube['ef']):
            frame_name = self.frame_name_format.format(frame_id)
            frame_path = join(video_path, frame_name)
            frame_paths.append(frame_path)

        boxes = tube['boxes'].astype('float32')
        # TODO: function modifies boxes inplace, not a big fan of this
        self._convert_boxes(boxes)

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

    # Data Analysis Methods
    # TODO: Turn some of these into properties?

    def get_tube_lengths(self):
        lengths = []
        for tube in self.tubes:
            lengths.append(len(tube))
        return lengths

    def get_average_tube_length(self):
        lengths = self.get_tube_lengths()
        return sum(lengths) / len(lengths)

    def get_max_tube_length(self):
        lengths = self.get_tube_lengths()
        return max(lengths)


def main():
    dataset = BoundingBoxDataset(
        '/mnt/hdd/datasets/ucf24',
        'rgb-images',
        'splitfiles/pyannot.pkl',
        'splitfiles/testlist01.txt',
        transform=ImglistToTensor(dim=0)
    )

    # frames, progress_values = dataset[0]
    print(len(dataset))
    print(dataset.get_average_tube_length(), dataset.get_max_tube_length())


if __name__ == '__main__':
    main()
