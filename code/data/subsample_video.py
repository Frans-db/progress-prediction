import argparse
import os
from os.path import join
import multiprocessing
import subprocess
import shutil

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='/mnt/hdd/datasets/cholec80')
    parser.add_argument('--save_root', type=str, default='/home/frans/Datasets/cholec80')

    return parser.parse_args()

def subsample_video(video_dir: str, save_dir: str):
    if os.path.isdir(save_dir):
        return

    print(f'[subsampling {video_dir}]')
    os.mkdir(save_dir)
    frames = sorted(os.listdir(video_dir))[::25]
    frame_paths = [join(video_dir, frame_name) for frame_name in frames]

    # Switching from subprocess to shutil here
    # linux seems to have problems with cp/mv many files at once
    # using shutfil for individual files appears to be much faster
    for frame_path in frame_paths:
        shutil.copy(frame_path, save_dir)


def main():
    args = parse_arguments()

    video_data_root = join(args.data_root, 'frames')
    video_save_root = join(args.save_root, 'frames_subsampled')

    # create save directories
    if not os.path.isdir(video_save_root):
        os.mkdir(video_save_root)
    
    # get all video directories
    split_arguments = []
    video_names = sorted([video_name for video_name in os.listdir(video_data_root)])

    for video_name in video_names:
        video_dir = join(video_data_root, f'{video_name}')
        video_save_dir = join(video_save_root, video_name)
        split_arguments.append((video_dir, video_save_dir))

    with multiprocessing.Pool(1) as pool:
        pool.starmap(subsample_video, split_arguments)


if __name__ == '__main__':
    main()