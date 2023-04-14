import os
from typing import List

from torch.utils.data import Dataset

class ProgressBaseDataset(Dataset):
    def __init__(self, root: str, data_type: str, split_file: str) -> None:
        super(Dataset, self).__init__()
        self.root = root
        self.data_root = os.path.join(self.root, data_type)

        split_file_path = os.path.join(self.root, 'splitfiles', split_file)
        self.split_names = self._load_split_file(split_file_path)

    @staticmethod
    def _load_split_file(path: str) -> List[str]:
        with open(path, 'r') as f:
            data = f.readlines()
        return [row.strip() for row in data]

class ProgressDataset(ProgressBaseDataset):
    def __init__(self, root: str, data_type: str, split_file: str) -> None:
        super(ProgressDataset, self).__init__(root, data_type, split_file)
        self._load_data(self.split_names, self.data_root)

    @staticmethod
    def _load_data(split_files: List[str], data_root: str):
        data, progress = [], []
        for filename in split_files:
            path = os.path.join(data_root, f'{filename}.txt')
            with open(path, 'r') as f:
                lines = f.readlines()
            video_data = []
            for line in lines:
                print(list(map(lambda x: float(x.strip()), line.split(' '))))

# class Progress2DDataset(ProgressDataset):
#     def __init__(self, root, data_type, split_file):
#         pass

if __name__ == '__main__':
    dataset = ProgressDataset('/home/frans/Datasets/breakfast', 'features/dense_trajectories', 'train_s1.txt')