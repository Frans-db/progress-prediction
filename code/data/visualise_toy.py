import argparse
import pickle
import os

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='/mnt/hdd/datasets/')
    parser.add_argument('--dataset', type=str, default='toy')
    parser.add_argument('--video_name', type=str, default='00000')

    return parser.parse_args()

def load_pickle(file_path: str):
    with open(annotation_path, 'rb') as f:
        data = pickle.load(f)
    return data

def main():
    args = parse_arguments()

    dataset_path = os.path.join(args.data_root, args.dataset)
    frames_dir = os.path.join(dataset_path, 'rgb-images', args.video_name)
    annotation_path = os.path.join(dataset_path, 'splitfiles/pyannot.pkl')

    frame_names = os.listdir(frames_dir)
    annotations = load_pickle(annotation_path)
    print(annotations[args.video_name])

if __name__ == '__main__':
    main()