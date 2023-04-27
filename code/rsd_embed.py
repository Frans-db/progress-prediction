import torch
import torch.nn as nn
import torchvision.transforms as transforms
import argparse
import os
from PIL import Image
from tqdm import tqdm

from networks import RSDFlat

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='/home/frans/Datasets/')
    parser.add_argument('--dataset', type=str, default='cholec80')
    parser.add_argument('--data_type', type=str, default='rgb-images')

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
        transforms.Resize((224, 224))
    ])

    paths = []
    video_names = sorted(os.listdir(root))
    for video_name in tqdm(video_names):
        video_path = os.path.join(root, video_name)
        frame_names = sorted(os.listdir(video_path))

        embeddings = []
        for frame_name in frame_names:
            frame_path = os.path.join(video_path, frame_name)

            frame = Image.open(frame_path)
            frame = transform(frame).unsqueeze(dim=0).to(device)
            embedded = model.resnet(frame).squeeze().cpu().tolist()
            embeddings.append(' '.join(map(str, embedded)))
        
        target_path = os.path.join(target, f'{video_name}.txt')
        with open(target_path, 'w+') as f:
            f.write('\n'.join(embeddings))

    # for root, dirs, files in os.walk(root, topdown=False):
    #     for name in files:
    #         path = os.path.join(root, name)
    #         paths.append(path)
    # for path in tqdm(paths):
        # frame = Image.open(path)
        # frame = transform(frame).unsqueeze(dim=0).to(device)
        # embedded = model.resnet(frame)

    #     print(path)
        # print(embedded.shape)
        # print(frame.shape)



if __name__ == '__main__':
    main()