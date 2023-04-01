import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import pickle

class UCFDataset(Dataset):
    def __init__(self, root: str, data_type: str, split_path: str, transform=None, sample_transform=None, image_size = (320, 240)) -> None:
        super(UCFDataset, self).__init__()
        self.root = root
        self.data_root = os.path.join(root, data_type)

        self.split_files = self._load_split(os.path.join(root, split_path))


        self.transform = transform
        self.sample_transform = sample_transform

    def __getitem__(self, index: int) -> tuple[torch.FloatTensor, torch.FloatTensor]:
        pass

    def __len__(self) -> int:
        return len(self.split_files)

    def _load_split(self, split_path: str) -> list[str]:
        with open(split_path, 'r') as f:
            lines = f.readlines()
        return [line.strip() for line in lines]

    def _load_tubes(self, annotation_path: str):
        with open(annotation_path, 'rb') as f:
            database = pickle.load(f)
        
        for video_name in database:
            if video_name not in self.split_files:
                continue

            video_path = join(self.data_root, video_name)
            tubes = database[video_name]['annotations']

            for tube in tubes:
                frame_paths, boxes = self._load_tube(tube, video_path)

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
