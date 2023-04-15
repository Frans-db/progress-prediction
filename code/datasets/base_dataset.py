from torch.utils.data import Dataset
import os
from typing import List

class BaseDataset(Dataset):
    def __init__(self, root: str, data_type: str, split_file: str, transform = None, sample_transform = None) -> None:
        super(Dataset, self).__init__()
        self.root = root
        self.data_root = os.path.join(self.root, data_type)

        split_file_path = os.path.join(self.root, 'splitfiles', split_file)
        self.split_names = self._load_split_file(split_file_path)

        self.transform = transform
        self.sample_transform = sample_transform

    @staticmethod
    def _load_split_file(path: str) -> List[str]:
        with open(path, 'r') as f:
            data = f.readlines()
        return [row.strip() for row in data]