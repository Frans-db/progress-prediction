import argparse
from typing import List
import random
import torch
from torchvision import transforms
import os
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import make_grid
from torchvision.io import read_image
import subprocess

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--root', type=str, default='/home/frans/Datasets/')
    parser.add_argument('--save_dir', type=str, default='bars')

    parser.add_argument('--num_videos', type=int, default=1000)
    parser.add_argument('--actions_per_video', type=int, default=4)
    parser.add_argument('--notches_per_action', type=int, default=4)

    parser.add_argument('--min_frames_per_notch', type=int, default=1)
    parser.add_argument('--max_frames_per_notch', type=int, default=1)
    parser.add_argument('--min_video_multiplier', type=int, default=1)
    parser.add_argument('--max_video_multiplier', type=int, default=1)

    parser.add_argument('--notch_width', type=int, default=2)
    parser.add_argument('--notch_height', type=int, default=32)
    parser.add_argument('--video_size', type=int, default=32)

    parser.add_argument('--visualise', action='store_true')

    return parser.parse_args()

# R G B
COLOURS = torch.Tensor([(219, 0, 91), (73, 66, 228), (208, 245, 190)]) / 255

def create_frame(num_notces, colour, notch_width: int, notch_height: int, size: int) -> torch.Tensor:
    frame = torch.zeros(3, size, size)
    for i in range(num_notces):
        frame[:, 0:notch_height, i*notch_width:(i+1)*notch_width] = torch.Tensor(colour)[:, None, None]
    return frame

def lerp(value, start, end):
    return start * (1 - value) + end * value

def lerp_iterables(value, start, end):
    new_iter = []
    for (start_val, end_val) in zip(start, end):
        new_iter.append(lerp(value, start_val, end_val))
    return new_iter

def lerp_colours(value: float, color1, color2, color3):
    if value < 0.5:
        return lerp_iterables(value / 0.5, color1, color2)
    else:
        return lerp_iterables((value - 0.5) / 0.5, color2, color3)

def create_data(args):
    save_dir = os.path.join(args.root, args.save_dir)
    os.mkdir(save_dir)
    os.mkdir(os.path.join(save_dir, 'rgb-images'))
    os.mkdir(os.path.join(save_dir, 'splitfiles'))
    transform = transforms.ToPILImage()

    notches_per_video = args.actions_per_video * args.notches_per_action

    for video_index in tqdm(range(args.num_videos)):
        video_dir = os.path.join(save_dir, 'rgb-images', f'{video_index:05d}')
        os.mkdir(video_dir)
        if args.min_video_multiplier == args.max_video_multiplier:
            video_multiplier = args.min_video_multiplier
        else:
            video_multiplier = random.randint(args.min_video_multiplier, args.max_video_multiplier)
        num_notches = 0
        frame_index = 0
        for action_index in range(args.actions_per_video):
            for notch_index in range(args.notches_per_action):
                if args.min_frames_per_notch == args.max_frames_per_notch:
                    notch_length = args.min_frames_per_notch * video_multiplier
                else:
                    notch_length = random.randint(args.min_frames_per_notch, args.max_frames_per_notch) * video_multiplier
                num_notches += 1
                colour = lerp_colours(num_notches / notches_per_video, COLOURS[0], COLOURS[1], COLOURS[2])
                frame_array = create_frame(num_notches, colour, args.notch_width, args.notch_height, args.video_size)
                frame = transform(frame_array)

                for _ in range(notch_length):
                    frame_path = os.path.join(video_dir, f'{frame_index:05}.jpg')
                    frame.save(frame_path)
                    frame_index += 1

    video_names = [f'{video_id:05d}\n' for video_id in range(args.num_videos)]
    random.shuffle(video_names)
    with open(os.path.join(save_dir, 'splitfiles', 'train.txt'), 'w+') as f:
        f.writelines(sorted(video_names[:int(0.9 * args.num_videos)]))
    with open(os.path.join(save_dir, 'splitfiles', 'test.txt'), 'w+') as f:
        f.writelines(sorted(video_names[int(0.9 * args.num_videos):]))


def visualise(args):
    data_dir = os.path.join(args.root, args.save_dir, 'rgb-images')
    video_names = os.listdir(data_dir)[:16]

    frames_per_video = {}
    max_num_frames = 0
    for video_name in video_names:
        video_path = os.path.join(data_dir, video_name)
        num_frames = len(os.listdir(video_path))

        frames_per_video[video_name] = num_frames
        max_num_frames = max(num_frames, max_num_frames)

    for i in tqdm(range(max_num_frames)):
        frames = []
        for video_name in video_names:
            frame_index = min(i, frames_per_video[video_name] - 1)
            frame_path = os.path.join(data_dir, video_name, f'{frame_index:05d}.jpg')
            frames.append(read_image(frame_path))
        grid = make_grid(frames, nrow=4, padding=2, pad_value=255)
        plt.figure(figsize=(8, 8))
        plt.imshow(np.transpose(grid, [1, 2, 0]))
        plt.axis('off')
        plt.savefig(f'./plots/bars/{i:03d}.jpg')
        plt.clf()

    subprocess.call([
        'ffmpeg', '-framerate', '3', '-i', './plots/bars/%03d.jpg', '-r', '30', '-pix_fmt', 'yuv420p',
        './plots/bars/out.mp4'
    ])

def main():
    args = parse_args()
    random.seed(args.seed)
    if args.visualise:
        visualise(args)
    else:
        create_data(args)

  

if __name__ == '__main__':
    main()