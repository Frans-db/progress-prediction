import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import argparse
import os
import pickle

from networks import InceptionI3d
from datasets import ChunkDataset, load_splitfile
from experiment import get_device, set_seeds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--model_path", type=str, default="/home/frans/Datasets/models/rgb_imagenet.pth"
    )
    parser.add_argument("--root", type=str, default="/home/frans/Datasets/")
    parser.add_argument("--dataset", type=str, default="breakfast")
    parser.add_argument(
        "--save_dir",
        type=str,
        default="features/i3d_embeddings",
    )
    parser.add_argument("--splitfile", type=str, default="all.txt")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--subsample_fps", type=int, default=1)

    return parser.parse_args()


def main():
    args = parse_args()
    device = get_device()
    set_seeds(args.seed)
    data_root = os.path.join(args.root, args.dataset)

    network = InceptionI3d(400, in_channels=3)
    network.load_state_dict(torch.load(args.model_path))
    network.train(False)
    network.to(device)

    data = []
    if args.splitfile.endswith(".txt"):
        split_path = os.path.join(data_root, "splitfiles", args.splitfile)
        splitnames = load_splitfile(split_path)
        for video_name in tqdm(splitnames):
            video_path = os.path.join(data_root, "rgb-images", video_name)
            frame_paths = [
                os.path.join(video_path, frame_name)
                for frame_name in sorted(os.listdir(video_path))
            ][::args.subsample_fps]
            data.append((video_name, frame_paths))
    elif args.splitfile.endswith(".pkl"):
        pickle_path = os.path.join(data_root, "splitfiles", args.splitfile)
        with open(pickle_path, "rb") as f:
            database = pickle.load(f)
        for video_name in tqdm(database):
            annotations = database[video_name]["annotations"]
            for tube_index, tube in enumerate(annotations):
                frame_paths = []
                for frame_index in range(tube["sf"], tube["ef"]):
                    frame_name = f"{frame_index+1:05d}.jpg"
                    frame_path = os.path.join(
                        data_root, "rgb-images", video_name, frame_name
                    )
                    frame_paths.append(frame_path)
                data.append((f"{video_name}_{tube_index}", frame_paths))

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )
    for video_name, frame_paths in tqdm(data):
        dataset = ChunkDataset(frame_paths, 15, transform=transform)
        dataloader = DataLoader(
            dataset, batch_size=args.batch_size, num_workers=4, shuffle=False
        )

        with torch.no_grad():
            embedding_texts = []
            for frames in dataloader:
                frames = frames.to(device)
                embeddings = network.extract_features(frames)
                embeddings = torch.flatten(embeddings, start_dim=1).tolist()
                for embedding in embeddings:
                    embedding_texts.append(" ".join(map(str, embedding)))
            save_path = os.path.join(
                args.root, args.dataset, args.save_dir, f"{video_name}.txt"
            )
            if "/" in save_path:
                directories = "/".join(save_path.split("/")[:-1])
                try:
                    os.makedirs(directories)
                except FileExistsError:
                    pass
            with open(save_path, "w+") as f:
                f.write("\n".join(embedding_texts))


if __name__ == "__main__":
    main()
