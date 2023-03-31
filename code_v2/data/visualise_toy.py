import argparse
import pickle
import os
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='/mnt/hdd/datasets/')
    parser.add_argument('--dataset', type=str, default='toy')
    parser.add_argument('--video_name', type=str, default='00000')

    parser.add_argument('--save_dir', type=str,
                        default='/home/frans/Downloads/temp')
    parser.add_argument('--frame_format', type=str,
                        default='frame_{0:03d}.png')

    return parser.parse_args()


def load_pickle(file_path: str):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data


def get_video_boxes(annotations, video_name: str):
    video_annotations = annotations[video_name]['annotations']
    # toy dataset always has a single tube so we can safely get the boxes from index 0
    return video_annotations[0]['boxes']


def setup_paths(args):
    dataset_path = os.path.join(args.data_root, args.dataset)
    frames_dir = os.path.join(dataset_path, 'rgb-images', args.video_name)
    annotation_path = os.path.join(dataset_path, 'splitfiles/pyannot.pkl')
    if not os.path.isdir(args.save_dir):
        os.mkdir(args.save_dir)
    frame_save_path = os.path.join(args.save_dir, args.frame_format)

    return dataset_path, frames_dir, annotation_path, frame_save_path


def main():
    args = parse_arguments()
    dataset_path, frames_dir, annotation_path, frame_save_path = setup_paths(
        args)

    annotations = load_pickle(annotation_path)

    frame_names = sorted(os.listdir(frames_dir))
    boxes = get_video_boxes(annotations, args.video_name)

    for i, (frame_name, box) in enumerate(zip(frame_names, boxes)):
        frame_path = os.path.join(frames_dir, frame_name)
        frame = Image.open(frame_path).convert('RGB')

        fig, (ax1) = plt.subplots()

        ax1.imshow(frame)
        rect = patches.Rectangle(
            (box[0], box[1]), box[2], box[3], linewidth=2, edgecolor='r', facecolor='none')
        ax1.add_patch(rect)
        ax1.set_title(f'box {box[0], box[1], box[2], box[3]}')

        plt.savefig(frame_save_path.format(i))
        plt.close()


if __name__ == '__main__':
    main()
