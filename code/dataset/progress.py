import os
import torch
from PIL import Image

class ProgressDataset(torch.utils.data.Dataset):
    def __init__(self, 
                 root_path: str,
                 transform = None) -> None:
        super().__init__()
        self.root_path = root_path
        self.items = os.listdir(self.root_path)
        self.transform = transform
        
    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        item_directory = os.path.join(self.root_path, item)
        image_files = sorted(os.listdir(item_directory))
        num_images = len(image_files)
        

        images = []
        labels = []
        for i,filename in enumerate(image_files):
            image_path = os.path.join(item_directory, filename)
            images.append(Image.open(image_path))
            labels.append((i+1) / num_images)

        if self.transform is not None:
            return self.transform(images), labels
        return images, labels
            
    def _load_image(self, path: str) -> Image.Image:
        return Image.open(path).convert('RGB')