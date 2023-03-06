import argparse
import os
from os.path import join
import multiprocessing
import subprocess

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='/mnt/hdd/datasets/cholec80')
    parser.add_argument('--save_root', type=str, default='/mnt/hdd/datasets/cholec80_split')

    return parser.parse_args()

def split_video(video_path: str, save_dir: str):
    if os.path.isdir(save_dir):
        return

    print(f'[splitting {video_path}]')
    os.mkdir(save_dir)
    frame_save_path = join(save_dir, 'frame_%06d.jpg')
    subprocess.run(
        ['ffmpeg', '-i', video_path, frame_save_path]
    )

def main():
    args = parse_arguments()

    video_root = join(args.data_root, 'videos')
    video_save_root = join(args.save_root, 'videos')
    video_temp_root = join(args.save_root, 'temp')

    # create save directories
    if not os.path.isdir(args.save_root):
        os.mkdir(args.save_root)
    if not os.path.isdir(video_save_root):
        os.mkdir(video_save_root)
    if not os.path.isdir(video_temp_root):
        os.mkdir(video_temp_root)
    
    # get all mp4's from the data root
    split_arguments = []
    video_names = sorted([video_name[:-4] for video_name in os.listdir(video_root) if 'mp4' in video_name])

    for video_name in video_names:
        video_path = join(video_root, f'{video_name}.mp4')
        video_save_dir = join(video_temp_root, video_name)
        split_arguments.append((video_path, video_save_dir))

    with multiprocessing.Pool(1) as pool:
        pool.starmap(split_video, split_arguments)


if __name__ == '__main__':
    main()