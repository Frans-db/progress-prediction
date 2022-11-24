# Author: Silvia-Laura Pintea

import argparse
import matplotlib.pyplot as plt
import torchvision
import torch
from torch.utils.data import Subset, DataLoader
from typing import List, Union, Tuple, Any
import random
from PIL import Image
import numpy as np
import itertools
import math
import pickle
from mycolorpy import colorlist as mcp
import os
import sys

# sys.path.append("../../")
# from util.utils import (
#     check_rootfolders,
# )

parser = argparse.ArgumentParser()

# ------------------------ Dataset Configs ------------------------
parser.add_argument("--seed", type=int, default=0, help="Manual seed")
parser.add_argument(
    "--path",
    type=str,
    default="./data/MNIST/original/",
    help="Dataset location ...",
)
parser.add_argument(
    "--name",
    type=str,
    default="mnist_d0",
    help="datasetname",
)

parser.add_argument(
    "--bg_path",
    type=str,
    default="./data/EPIC-KITCHENS/train/P01/rgb_frames/",
    help="Frames to use for bg",
)
parser.add_argument(
    "--new_path", type=str, default="./data/MNIST/toy/", help="New dataset path ..."
)
parser.add_argument(
    "--noise_var", type=float, default=1, help="Noise variance to trajectory in pixels"
)
parser.add_argument(
    "--step", type=float, default=5, help="The step size with which digits move"
)
parser.add_argument(
    "--shuffle", type=float, default=0, help="Percentage of tasks to shuffle around" # 20
)

parser.add_argument("--drop", type=float, default=0, help="Percentage of tasks to drop") # 10

parser.add_argument("--repeated", type=int, default=-1, help="Which task is repeated") # 3
parser.add_argument("--repeats", type=int, default=0, help="Maximum repeats: 3") # 3

parser.add_argument(
    "--max-segment", type=int, default=10, help="The maximum segment=task length" # 10
)
parser.add_argument(
    "--min-segment", type=int, default=10, help="The minimum segment=task length" # 10
)

parser.add_argument(
    "--min-speed", type=float, default=3.0 / 4.0, help="The minimum speed scaling"
)
parser.add_argument(
    "--max-speed", type=float, default=3.0, help="The maximum speed scaling"
)


args = parser.parse_args()


def create_activity_mnist(
    train: bool, targets: List, file_name: str = "activity_mnist"
):
    global args
    random.seed(args.seed)
    np.random.seed(args.seed)

    # ---------------------
    dataset = torchvision.datasets.MNIST(args.path, train=train, download=True)
    subsets = {
        target: Subset(dataset, [i for i, (x, y) in enumerate(dataset) if y == target])
        for _, target in dataset.class_to_idx.items()
    }

    subset_list = []
    min_len = math.inf
    for (targ, motion) in targets:
        subset_list.append((motion, targ, subsets[targ]))  # for every sub-task
        min_len = len(subsets[targ]) if len(subsets[targ]) < min_len else min_len

    for i in range(len(subset_list)):
        print("#images for class ", i, ": ", len(subset_list[i][2]))

    videos = []
    labels = []
    for idx in range(0, 10):  # for every video
        # Define the activities
        new_subset_list = randomly_permute_drop_repeat_tasks(subset_list)

        # Pick a global speed scaling factor
        video_speed = (
            args.min_speed + (args.max_speed - args.min_speed) * 0.5 # random.random() removed random for now while playing around with dataloader
        )

        make_data(videos, labels, idx, new_subset_list, video_speed)
    assert len(videos) == len(labels)

    videos = add_background(videos)

    # Write down the data
    print("Created data: (", videos[0].shape, ",", labels[0].shape, ") x ", len(videos))
    
    '''
    The VideoFrameDataset expects the annotation file to be in the following format:
    video_id start_frame end_frame label* [as many labels as you want]
    Normally these labels are integer id's, but as a workaround I'm returning them as floats
    and each id i is the completion percentage of frame i. Note that this does not yet work
    with subsampling
    '''
    annotations_text = ''
    for video_id, video in enumerate(videos):
        # Create directory to store the video frames in
        os.mkdir(f'{args.new_path}/{video_id:05d}')
        # Add video_id, start_frame, end_frame to the annotations text
        annotations_text += f'{video_id:05d} {1} {len(video)} '
        percentages = []
        for frame_id, frame in enumerate(video):
            # Add a percentage done label for each frame
            percentages.append(f'{frame_id / len(video)}')
            # Save the frame
            # plt.imshow(frame)
            # plt.savefig(f'{args.new_path}/{video_id:05d}/img_{frame_id:05d}.png')
            # plt.clf()
            image = Image.fromarray(np.uint8(frame * 255)).convert('RGB')
            image.save(f'{args.new_path}/{video_id:05d}/img_{(frame_id+1):05d}.png')
        annotations_text += ' '.join(percentages)
        annotations_text += '\n'
            
    # Save the annotations txt in the data path root
    with open(f'{args.new_path}/annotations.txt', 'w+') as f:
        f.write(annotations_text)

    # with open(args.new_path + file_name + ".pkl", "wb+") as f:
    #     pickle.dump((videos, labels), f)

    # See some examples
    # view_videos(videos, labels)


def add_background(videos):
    global args

    dirs = [
        os.path.join(args.bg_path, d)
        for d in os.listdir(args.bg_path)
        if os.path.isdir(os.path.join(args.bg_path, d))
    ]
    print("Background dirs:", dirs)

    frames = []
    for d in dirs:
        files = [os.path.join(d, f) for f in os.listdir(d) if f.endswith(".jpg")]
        frames.append(files)

    # Loop over videos:
    print("Videos: #", len(videos))
    for i, v in enumerate(videos):
        # pick randomly a background:
        bg_id = random.randint(0, len(frames) - 1)

        # pick randomly a start frame:
        start_fr = max(0, random.randint(0, len(frames[bg_id]) - v.shape[0]))
        assert start_fr + v.shape[0] <= len(frames[bg_id])

        for f in range(0, v.shape[0]):
            # Read the video frame
            frame_name = frames[bg_id][start_fr + f]
            im = np.array(Image.open(frame_name)) / 255.0

            # Resize to W,H of video
            startx = im.shape[0] // 2 - videos[0].shape[1] // 2
            endx = im.shape[0] // 2 + videos[0].shape[1] // 2
            starty = im.shape[1] // 2 - videos[0].shape[2] // 2
            endy = im.shape[1] // 2 + videos[0].shape[2] // 2
            """
            im = np.array(
                im.resize(
                    (videos[0].shape[1], videos[0].shape[2]),
                    Image.BILINEAR,
                )
            )
            """
            v[f, :, :, :] = (v[f, :, :, :] + im[startx:endx, starty:endy, :]) / 2.0

            # Show the frame
            # plt.imshow(v[f,:,:])
            # plt.savefig("tmp/frame"+str(f)+".jpg")
            # plt.pause(0.1)

        # plt.show()
        if i % 100 == 0:
            print("Processed ", i, " videos..")
    return videos


def randomly_permute_drop_repeat_tasks(subset_list: List):
    global args
    new_subset_list = subset_list.copy()

    # Cannot shuffle the first and last tasks
    # args.shuffle percentage of the tasks can be permuted
    if args.shuffle > 0:
        shuffle = np.random.rand()
        if shuffle < args.shuffle / 100.0:
            perm_size = math.ceil(args.shuffle / 100.0 * (len(subset_list) - 2))
            perms = [
                subset
                for subset in itertools.combinations(range(1, len(subset_list) - 1), 2)
            ]

            random.shuffle(perms)
            perms = perms[0:perm_size]

            for perm in perms:
                new_subset_list[perm[0]], new_subset_list[perm[1]] = (
                    new_subset_list[perm[1]],
                    new_subset_list[perm[0]],
                )

    # Allow for repetitions in the middle:
    if args.repeated < len(new_subset_list) - 1 and args.repeats > 1:
        args.repeated = int(args.repeated)
        to_repeat = new_subset_list[args.repeated]
        nr_repeats = random.randint(2, args.repeats)
        repeat_pos = np.arange(
            args.repeated - 2 * args.repeats, args.repeated + 2 * args.repeats + 2, 2
        ).tolist()
        inserts = 0
        for pos in repeat_pos:
            if (
                (pos > 0 and pos < len(new_subset_list) - 1)
                and (new_subset_list[pos - 1]) != to_repeat
                and (new_subset_list[pos + 1]) != to_repeat
            ):
                new_subset_list.insert(pos, to_repeat)
                inserts += 1
            if inserts >= nr_repeats:
                break

    # Cannot drop the first and last tasks
    if args.drop > 0:
        drop = np.random.rand()
        if drop < args.drop / 100.0:
            task_dropped = random.randint(1, len(new_subset_list) - 2)
            del new_subset_list[task_dropped]

    return new_subset_list


def get_motion(
    x: float, y: float, f: int, motion: str, shape: List, video_speed: float
):
    # video is sped/slowed between 0.75 and 3 so motion needs to be the other way
    step = args.step * 1.0 / video_speed

    if motion == "horizontal":
        x = x + step * f
    elif motion == "inv-horizontal":
        x = x - step * f
    elif motion == "vertical":
        y = y + step * f
    elif motion == "diagonal":
        x = x + math.sqrt(step * step) * f
        y = y + math.sqrt(step * step) * f
    elif motion == "inv_diagonal":
        x = x - math.sqrt(step * step) * f
        y = y + math.sqrt(step * step) * f

    # If the motion exceeds the image, circulate
    """
    if motion.startswith("inv"):
        x = int(x)
        if x <= 0:
            x = shape[1] + x
        y = max(0, int(y)) % shape[0]
    else:
        x = max(0, int(x)) % shape[1]
        y = max(0, int(y)) % shape[0]
    """

    # clip at min-max
    x = min(max(0, int(x)), shape[1])
    y = min(max(0, int(y)), shape[0])
    return x, y


def get_color(image, motion, f, frames):
    rgb_image = np.stack([image, image, image], axis=2)
    if motion == "horizontal":
        colorS = np.array([255.0, 0.0, 255.0])
        colorE = np.array([255.0, 255.0, 0.0])
        color = (colorE * (f + 1) / frames) + (colorS * (1 - (f + 1) / frames))
        rgb_image = rgb_image / 255.0 * color

    elif motion == "inv-horizontal":
        colorS = np.array([0.0, 0.0, 255.0])
        colorE = np.array([255.0, 0.0, 0.0])
        color = (colorE * (f + 1) / frames) + (colorS * (1 - (f + 1) / frames))
        rgb_image = rgb_image / 255.0 * color

    elif motion == "vertical":
        colorS = np.array([0.0, 255.0, 255.0])
        colorE = np.array([0.0, 255.0, 0.0])
        color = (colorE * (f + 1) / frames) + (colorS * (1 - (f + 1) / frames))
        rgb_image = rgb_image / 255.0 * color

    elif motion == "diagonal":
        colorS = np.array([255.0, 102.0, 0.0])
        colorE = np.array([204.0, 153.0, 255.0])
        color = (colorE * (f + 1) / frames) + (colorS * (1 - (f + 1) / frames))
        rgb_image = rgb_image / 255.0 * color

    elif motion == "inv_diagonal":
        colorS = np.array([153.0, 153.0, 255.0])
        colorE = np.array([153.0, 204.0, 0.0])
        color = (colorE * (f + 1) / frames) + (colorS * (1 - (f + 1) / frames))
        rgb_image = rgb_image / 255.0 * color

    return rgb_image / 255.0


def make_data(
    videos: List[np.array],
    labels: List[np.array],
    idx,
    subset_list: List["torchvision.datasets"],
    video_speed: float,
):
    global args
    # Loop over list of activities
    for (motion, targ, subset) in subset_list:

        # Read the pill image associated with this video activity
        (pil_image, _) = subset.__getitem__(idx)
        pil_image = pil_image.resize(
            (int(pil_image.size[0] * 0.5), int(pil_image.size[1] * 0.5)),
            Image.BILINEAR,
        )
        image = np.array(pil_image)
        imsize = image.shape

        # Select the number of frames for this activity
        frames = int(random.randint(args.min_segment, args.max_segment) * video_speed)
        try:
            start_frame = videos[idx].shape[0]
            videos[idx] = np.pad(
                videos[idx],
                ((0, frames), (0, 0), (0, 0), (0, 0)),
                "constant",
                constant_values=((0, 0), (0, 0), (0, 0), (0, 0)),
            )
            labels[idx] = np.pad(
                labels[idx], ((0, frames)), "constant", constant_values=((0, targ))
            )
        except:
            start_frame = 0
            videos.append(np.zeros((frames, imsize[0] * 3, imsize[1] * 3, 3)))
            labels.append(np.ones(frames) * targ)
            assert len(videos) == idx + 1

        # Define the starting point of the motion: within 5 px
        if motion.startswith("inv"):
            pick_start_x = random.randint(
                videos[idx].shape[2] - 5 - image.shape[1],
                videos[idx].shape[2] - image.shape[1],
            )
        else:
            pick_start_x = random.randint(0, 5)

        pick_start_y = random.randint(0, 5)

        for f in range(0, frames):
            start_x = (
                pick_start_x  # np.random.normal(loc=pick_start_x, scale=args.noise_var)
            )
            start_y = (
                pick_start_y  # np.random.normal(loc=pick_start_y, scale=args.noise_var)
            )
            start_x, start_y = get_motion(
                start_x,
                start_y,
                f + 1,
                motion,
                [
                    videos[idx].shape[1] - image.shape[0] - 1,
                    videos[idx].shape[2] - image.shape[1] - 1,
                ],
                video_speed,
            )

            rgb_image = get_color(image, motion, f, frames)
            # print(motion, start_x," : ",(start_x + image.shape[0])," ",start_y," : ",(start_y + image.shape[1])," ",videos[idx].shape)
            videos[idx][
                start_frame + f,
                start_y : (start_y + image.shape[0]),
                start_x : (start_x + image.shape[1]),
                :,
            ] = rgb_image


def view_videos(videos: List[np.array], labels: List[np.array]):

    fig, ax = plt.subplots(1, 2, constrained_layout=True)
    video_names = []
    class_length = []
    idx = []
    for i in range(0, 10):  # Over videos
        idx.append(random.randint(0, len(videos) - 1))

    # Get statistics plot
    max_cls = 0
    for i in range(0, len(idx)):  # Over 10 random videos
        class_length.append([])
        video_names.append("video" + str(i))

        cls_idx = 0
        unique_classes = []
        for f in range(0, videos[idx[i]].shape[0]):  # Over frames in video
            aclass = int(labels[idx[i]][f])
            unique_classes.append(aclass)
            try:  # We are still at the same class
                assert aclass == list(class_length[i][cls_idx - 1].keys())[0]
                class_length[i][cls_idx - 1][aclass] += 1
            except:
                class_length[i].append({aclass: 1})
                cls_idx += 1

        # Find the video that has all classes
        unique = np.unique(np.array(unique_classes)).tolist()
        if len(unique) > max_cls:
            max_cls = len(unique)
            class_names = unique

    class_names.sort()
    colors = mcp.gen_color(cmap="tab20", n=len(class_names))
    color_dict = {}
    for c in range(0, len(colors)):
        color_dict[class_names[c]] = colors[c]

    print("Class length: ", class_length)
    print("Color dictionary: ", color_dict)
    print("Video names: ", video_names)

    # Dummy plot to get the legend
    handles = [
        ax[0].barh(" ", 0, color=color_dict[cls], label=str(cls))[0]
        for cls in class_names
    ]

    # And then the real plot
    for i in range(0, len(idx)):  # loop over videos
        for a in range(0, len(class_length[i])):  # loop over activities
            cls = list(class_length[i][a].keys())[0]
            if a == 0:
                ax[0].barh(
                    video_names[i],
                    class_length[i][a][cls],
                    color=color_dict[cls],
                    label=str(cls),
                )
            else:
                ls = 0
                for p in range(0, a):  # loop over previous activities
                    prev_cls = list(class_length[i][p].keys())[0]
                    ls += class_length[i][p][prev_cls]
                ax[0].barh(
                    video_names[i],
                    class_length[i][a][cls],
                    left=ls,
                    color=color_dict[cls],
                    label=str(cls),
                )
    ax[0].legend(labels=class_names, handles=handles)

    i = 0
    # for i in range(0, len(idx)):  # loop over videos
    for f in range(0, videos[idx[i]].shape[0]):
        ax[1].imshow(videos[idx[i]][f, :, :])
        ax[1].set_title(
            "Video "
            + str(i)
            + ": Class"
            + str(labels[idx[i]][f])
            + " | Length "
            + str(videos[idx[i]].shape[0])
        )
        print(
            args.new_path
            + "frames-"
            + args.name
            + "/video_{:05d}-img_{:05d}.png".format(i, f)
        )
        plt.savefig(
            args.new_path
            + "frames-"
            + args.name
            + "/video_{:05d}-img_{:05d}.png".format(i, f)
        )
        # plt.pause(0.1)
    # plt.show()


def data_stats(videos: List[np.array], labels: List[np.array]):
    # Get statistics plot
    histo = {}
    cls_stats = {}
    avg_video = []
    max_frames = 0
    for i in range(0, len(videos)):  # Over all videos
        if max_frames < videos[i].shape[0]:
            max_frames = videos[i].shape[0]

    for i in range(0, len(videos)):  # Over all videos
        for f in range(0, videos[i].shape[0]):  # Over frames in video
            aclass = int(labels[i][f])
            try:  # We are still at the same class
                histo[aclass][f] += 1
            except:
                try:  # If the frame is not there yet
                    histo[aclass].append([0] * max_frames)
                    histo[aclass][f] = 1
                except:  # if the class is not there yet
                    histo[aclass] = [0] * max_frames
                    histo[aclass][f] = 1
            try:
                cls_stats[aclass][i] += 1.0
            except:
                try:
                    cls_stats[aclass].append([0] * len(videos))
                    cls_stats[aclass][i] = 1.0
                except:
                    cls_stats[aclass] = [0] * len(videos)
                    cls_stats[aclass][i] = 1.0
        avg_video.append(videos[i].shape[0])

    class_names = list(histo.keys())
    class_names.sort()
    colors = mcp.gen_color(cmap="tab20", n=len(class_names))
    color_dict = {}
    for c in range(0, len(colors)):
        color_dict[class_names[c]] = colors[c]
    print("Color dictionary: ", color_dict)

    # And then the real plot
    fig, ax = plt.subplots(1, len(class_names), constrained_layout=True)
    video_mean = np.array(avg_video).mean()
    video_std = np.array(avg_video).std()
    plt.title(("Video length: {0:.2f}" + "+/-{1:.2f}").format(video_mean, video_std))
    print(("Video length: {0:.2f}" + "+/-{1:.2f}").format(video_mean, video_std))
    for c in range(0, len(class_names)):  # loop over classes
        cls = class_names[c]
        ax[c].barh(list(range(0, len(histo[cls]))), histo[cls], color=color_dict[cls])

        cls_mean = np.array(cls_stats[cls]).mean()
        cls_std = np.array(cls_stats[cls]).std()
        cls_legend = ("Class " + str(cls) + ": {0:.2f}" + "+/-{1:.2f}").format(
            cls_mean, cls_std
        )
        ax[c].legend([cls_legend])
        print(
            ("Class " + str(cls) + ": {0:.2f}" + "+/-{1:.2f}").format(cls_mean, cls_std)
        )
        plt.savefig(args.new_path + "/stats/" + "img_{:05d}.png".format(f))
        plt.pause(0.1)
    plt.show()


def read_activity_mnist(name="activity_mnist_small"):
    with open(args.new_path + name + ".pkl", "rb") as f:
        (videos, labels) = pickle.load(f)

    print("Created data: (", videos[0].shape, ",", labels[0].shape, ") x ", len(videos))
    print(videos[0][0, :, :].max(), videos[0][0, :, :].min())

    # See some examples
    view_videos(videos, labels)

    # Get data statistics
    # data_stats(videos, labels)


# ------------------------------------------------------------------------------------
if __name__ == "__main__":
    
    # check_rootfolders(args.path, "raw")
    # check_rootfolders(args.new_path, "frames-" + args.name)
    create_activity_mnist(
        train=False,
        targets=[
            (1, "horizontal"),
            (3, "inv_diagonal"),
            (5, "inv-horizontal"),
            (7, "diagonal"),
            (9, "vertical"),
        ],
        file_name=args.name,
    )
    # read_activity_mnist(name=args.name)
