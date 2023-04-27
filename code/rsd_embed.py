import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torchvision.transforms as transforms
import argparse
import os
from PIL import Image
from tqdm import tqdm
from typing import List

from networks import RSDFlat

class BasicData(Dataset):
    def __init__(self, paths: List[str], transform = None) -> None:
        self.paths = paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]

        frame = Image.open(path)
        if self.transform:
            frame = self.transform(frame)

        return frame

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='/home/frans/Datasets/')
    parser.add_argument('--dataset', type=str, default='cholec80')
    parser.add_argument('--data_type', type=str, default='rgb-images')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=4)

    parser.add_argument('--load_experiment', type=str)
    parser.add_argument('--load_iteration', type=int)
    parser.add_argument('--target_directory', type=str)

    parser.add_argument('--backbone', type=str, default='resnet152')
    parser.add_argument('--backbone_name', type=str, default=None)
    parser.add_argument('--backbone_channels', type=int, default=512)

    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    root = os.path.join(args.data_root, args.dataset, args.data_type)
    target = os.path.join(args.data_root, args.dataset, args.target_directory)
    if not os.path.isdir(target):
        os.mkdir(target)

    model = RSDFlat(args, device).to(device)
    model.resnet.fc = nn.Identity()
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224), antialias=True)
    ])

    paths = []
    video_names = sorted(os.listdir(root))
    with torch.no_grad():
        for video_name in tqdm(video_names):
            video_path = os.path.join(root, video_name)
            frame_names = sorted(os.listdir(video_path))
            frame_paths = [os.path.join(video_path, frame_name) for frame_name in frame_names]
            dataset = BasicData(frame_paths, transform=transform)
            loader = DataLoader(dataset, num_workers=args.num_workers, batch_size=args.batch_size)

            embeddings = []
            for frames in loader:
                frames = frames.to(device)
                embedded = model.resnet(frames).cpu().tolist()
                for embedding in embedded:
                    embeddings.append(' '.join(map(str, embedding)))
        
            target_path = os.path.join(target, f'{video_name}.txt')
            with open(target_path, 'w+') as f:
                f.write('\n'.join(embeddings))



if __name__ == '__main__':
    main()