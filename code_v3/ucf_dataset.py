import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import pickle
from typing import List

class UCFDataset(Dataset):
    def __init__(self, root: str, data_type: str, split_path: str, annotation_path: str, transform=None, sample_transform=None, image_size = (320, 240)) -> None:
        super(UCFDataset, self).__init__()
        self.root = root
        self.data_root = os.path.join(root, data_type)

        self.transform = transform
        self.sample_transform = sample_transform
        self.image_size = image_size

        self.split_files = self._load_split(os.path.join(root, split_path))
        self.video_names, self.frame_paths, self.tubes = self._load_tubes(os.path.join(root, annotation_path))
    
    @property
    def lengths(self) -> List[int]:
        return [len(paths) for paths in self.frame_paths]

    @property
    def average_length(self) -> float:
        lengths = self.lengths
        return sum(lengths) / len(lengths)
    
    def __getitem__(self, index: int) -> tuple[torch.FloatTensor, torch.FloatTensor]:
        frame_paths = self.frame_paths[index]
        video_name = self.video_names[index]

        num_frames = len(frame_paths)
        progress_values = [(i+1) / num_frames for i in range(num_frames)]

        frames = []
        for frame_path in frame_paths:
            frame = Image.open(frame_path)
            if self.transform:
                frame = self.transform(frame)
            frames.append(frame)

        return video_name, torch.stack(frames), torch.FloatTensor(progress_values)

    def __len__(self) -> int:
        return len(self.tubes)


    def _load_split(self, split_path: str):
        with open(split_path, 'r') as f:
            lines = f.readlines()
        return [line.strip() for line in lines]

    def _load_tubes(self, annotation_path: str):
        with open(annotation_path, 'rb') as f:
            database = pickle.load(f)
        
        all_frame_paths = []
        all_boxes = []
        video_names = []
        for video_name in database:
            if video_name not in self.split_files:
                continue

            video_path = os.path.join(self.data_root, video_name)
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
            frame_path = os.path.join(video_path, frame_name)
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
