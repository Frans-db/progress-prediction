# Author: Silvia-Laura Pintea

import argparse
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms
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
from tqdm import tqdm

from pyramid_pooling import SpatialPyramidPooling

# sys.path.append("../../")
# from util.utils import (
#     check_rootfolders,
# )

parser = argparse.ArgumentParser()

# ------------------------ Dataset Configs ------------------------
parser.add_argument("--seed", type=int, default=42, help="Manual seed")
parser.add_argument(
    "--path",
    type=str,
    default="/home/frans/Datasets/",
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
    default="/home/frans/Datasets/EPIC-KITCHENS/train/P01/rgb_frames/",
    help="Frames to use for bg",
)
parser.add_argument(
    '--add_bg',
    action='store_true'
)
parser.add_argument(
    "--dataset", type=str, default="toy", help="New dataset name ..."
)
parser.add_argument(
    "--noise_var", type=float, default=1, help="Noise variance to trajectory in pixels"
)
parser.add_argument(
    "--step", type=float, default=5, help="The step size with which digits move"
)
parser.add_argument(
    "--shuffle", type=float, default=0, help="Percentage of tasks to shuffle around"  # 20
)

parser.add_argument("--drop", type=float, default=0,
                    help="Percentage of tasks to drop")  # 10

parser.add_argument("--repeated", type=int, default=-1,
                    help="Which task is repeated")  # 3
parser.add_argument("--repeats", type=int, default=0,
                    help="Maximum repeats: 3")  # 3

parser.add_argument("--big", type=int, default=5,
                    help="Which task to make big")  # 5
parser.add_argument("--big_multiplier", type=int, default=5,
                    help="How much bigger to make task")  # 5
parser.add_argument("--big_chance", type=float, default=0,
                    help="Percentage of tasks to make bigger")  # 100

parser.add_argument(
    "--max-segment", type=int, default=5, help="The maximum segment=task length"  # 10
)
parser.add_argument(
    "--min-segment", type=int, default=5, help="The minimum segment=task length"  # 5
)

parser.add_argument(
    "--min-speed", type=float, default=2, help="The minimum speed scaling"  # 3 / 4
)
parser.add_argument(
    "--max-speed", type=float, default=2, help="The maximum speed scaling"  # 3
)


args = parser.parse_args()


def create_activity_mnist(
    train: bool, targets: List, file_name: str = "activity_mnist"
):
    global args
    print(args)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # ---------------------
    dataset = torchvision.datasets.MNIST(args.path, train=train, download=True)
    subsets = {
        target: Subset(
            dataset, [i for i, (x, y) in enumerate(dataset) if y == target])
        for _, target in dataset.class_to_idx.items()
    }

    print(((args.min_segment+args.max_segment)/2)*((args.min_speed+args.max_speed)/2))

    videos = []
    labels = []
    database = {}

    for target in targets:
        subset_list = []
        min_len = math.inf
        for (targ, motion, colours) in target:
            subset_list.append((motion, colours, targ, subsets[targ]))  # for every sub-task
            min_len = len(subsets[targ]) if len(
                subsets[targ]) < min_len else min_len

        for i in range(len(subset_list)):
            print("#images for class ", i, ": ", len(subset_list[i][3]))

        
        idx_offset = len(videos)
        for idx in tqdm(range(0, min_len)):  # for every video
            # Define the activities
            new_subset_list = randomly_permute_drop_repeat_tasks(subset_list)

            # Pick a global speed scaling factor
            video_speed = (
                args.min_speed + (args.max_speed -
                                args.min_speed) * random.random()
            )

            make_data(videos, labels, database, idx + idx_offset, new_subset_list, video_speed)
        assert len(videos) == len(labels)

    if args.add_bg:
        videos = add_background(videos)

    # Write down the data
    print("Created data: (", videos[0].shape,
          ",", labels[0].shape, ") x ", len(videos))

    os.mkdir(f'{args.path}{args.dataset}')
    os.mkdir(f'{args.path}{args.dataset}/rgb-images')
    os.mkdir(f'{args.path}{args.dataset}/pooled')
    os.mkdir(f'{args.path}{args.dataset}/pooled/small')
    os.mkdir(f'{args.path}{args.dataset}/pooled/medium')
    os.mkdir(f'{args.path}{args.dataset}/pooled/large')
    os.mkdir(f'{args.path}{args.dataset}/labels')
    os.mkdir(f'{args.path}{args.dataset}/experiments')
    os.mkdir(f'{args.path}{args.dataset}/splitfiles')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transform = transforms.ToTensor()
    small_pool = SpatialPyramidPooling([1, 2]).to(device)
    medium_pool = SpatialPyramidPooling([1, 2, 3]).to(device)
    large_pool = SpatialPyramidPooling([1, 2, 4]).to(device)
    
    video_names = [f'{video_id:05d}\n' for video_id in range(len(videos))]
    random.shuffle(video_names)
    with open(os.path.join(args.path, args.dataset, 'splitfiles', 'trainlist01.txt'), 'w+') as f:
        f.writelines(sorted(video_names[:int(0.9 * len(videos))]))
    with open(os.path.join(args.path, args.dataset, 'splitfiles', 'testlist01.txt'), 'w+') as f:
        f.writelines(sorted(video_names[int(0.9 * len(videos)):]))

    with open(os.path.join(args.path, args.dataset, 'splitfiles/pyannot.pkl'), 'wb+') as f:
        pickle.dump(database, f)

    label_translations = []
    for video_id, (video, labels) in tqdm(enumerate(zip(videos, labels)), total=len(videos)):
        translated_labels = []
        for label in labels:
            label = int(label)
            if label not in label_translations:
                label_translations.append(label)
            translated_labels.append(str(label_translations.index(label)))

        with open(f'{args.path}{args.dataset}/labels/{video_id:05d}', 'w+') as f:
            f.write('\n'.join(translated_labels))

        os.mkdir(f'{args.path}{args.dataset}/rgb-images/{video_id:05d}')
        for frame_id, frame in enumerate(video):
            # Save the frame
            image = Image.fromarray(np.uint8(frame * 255)).convert('RGB')
            frame_name = f'{args.path}{args.dataset}/rgb-images/{video_id:05d}/{(frame_id+1):05d}.jpg'
            image.save(frame_name)

            transformed = transform(image).unsqueeze(dim=0).to(device)
            save_pool(transformed, small_pool, f'{args.path}{args.dataset}/pooled/small/{video_id:05d}.txt')
            save_pool(transformed, medium_pool, f'{args.path}{args.dataset}/pooled/medium/{video_id:05d}.txt')
            save_pool(transformed, large_pool, f'{args.path}{args.dataset}/pooled/large/{video_id:05d}.txt')

def save_pool(image, pool, path):
    pooled = pool(image).reshape(-1).tolist()
    pooled = map(str, pooled)
    text = '\n'.join(pooled)
    with open(path, 'w+') as f:
        f.write(text)



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
        files = [os.path.join(d, f)
                 for f in os.listdir(d) if f.endswith(".jpg")]
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
            v[f, :, :, :] = (
                v[f, :, :, :] + im[startx:endx, starty:endy, :]) / 2.0

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
            perm_size = math.ceil(args.shuffle / 100.0 *
                                  (len(subset_list) - 1))
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
        nr_repeats = random.randint(1, args.repeats)
        # print(nr_repeats)
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
            task_dropped = random.randint(1, len(new_subset_list) - 1)
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
    elif motion == "inv-diagonal":
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


def get_color(motion):
    if motion == "horizontal":
        colorS = np.array([255.0, 0.0, 255.0])
        colorE = np.array([255.0, 255.0, 0.0])
    elif motion == "inv-horizontal":
        colorS = np.array([0.0, 0.0, 255.0])
        colorE = np.array([255.0, 0.0, 0.0])
    elif motion == "vertical":
        colorS = np.array([0.0, 255.0, 255.0])
        colorE = np.array([0.0, 255.0, 0.0])
    elif motion == "diagonal":
        colorS = np.array([255.0, 102.0, 0.0])
        colorE = np.array([204.0, 153.0, 255.0])
    elif motion == "inv-diagonal":
        colorS = np.array([153.0, 153.0, 255.0])
        colorE = np.array([153.0, 204.0, 0.0])

    return colorS, colorE

def apply_color(image, colours, f, frames):
    rgb_image = np.stack([image, image, image], axis=2)
    colorS, colorE = colours
    color = (colorE * (f + 1) / frames) + (colorS * (1 - (f + 1) / frames))
    rgb_image = rgb_image / 255.0 * color

    return rgb_image / 255.0


def make_data(
    videos: List[np.array],
    labels: List[np.array],
    database: dict,
    idx,
    subset_list: List["torchvision.datasets"],
    video_speed: float,
):
    global args

    boxes = []
    numf = 0

    # Loop over list of activities
    for (motion, colours, targ, subset) in subset_list:
        # Read the pill image associated with this video activity
        (pil_image, _) = subset.__getitem__(idx % len(subset))
        pil_image = pil_image.resize(
            (int(pil_image.size[0] * 0.5), int(pil_image.size[1] * 0.5)),
            Image.BILINEAR,
        )
        image = np.array(pil_image)
        imsize = image.shape

        # Select the number of frames for this activity
        frames = int(random.randint(args.min_segment,
                     args.max_segment) * video_speed)
        if np.random.rand() < (args.big_chance / 100):
            frames *= args.big_multiplier

        numf += frames
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
                # np.random.normal(loc=pick_start_x, scale=args.noise_var)
                pick_start_x
            )
            start_y = (
                # np.random.normal(loc=pick_start_y, scale=args.noise_var)
                pick_start_y
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

            rgb_image = apply_color(image, colours, f, frames)
            # print(motion, start_x," : ",(start_x + image.shape[0])," ",start_y," : ",(start_y + image.shape[1])," ",videos[idx].shape)
            boxes.append([start_x, start_y, image.shape[1], image.shape[0]])
            videos[idx][
                start_frame + f,
                start_y: (start_y + image.shape[0]),
                start_x: (start_x + image.shape[1]),
                :,
            ] = rgb_image
    database[f'{idx:05d}'] = {
        "annotations": [
            {
                "label": 0,
                "sf": 0,
                "ef": numf,
                "boxes": np.array(boxes)
            },
        ],
        "numf": numf,
        "label": 0,
    }


def view_videos(videos: List[np.array], labels: List[np.array]):

    fig, ax = plt.subplots(1, 2, constrained_layout=True)
    video_names = []
    class_length = []
    idx = []
    for i in range(0, 10):  # Over videos
        idx.append(i)

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
        plt.pause(0.1)
    plt.show()


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
    plt.title(("Video length: {0:.2f}" +
              "+/-{1:.2f}").format(video_mean, video_std))
    print(("Video length: {0:.2f}" +
          "+/-{1:.2f}").format(video_mean, video_std))
    for c in range(0, len(class_names)):  # loop over classes
        cls = class_names[c]
        ax[c].barh(list(range(0, len(histo[cls]))),
                   histo[cls], color=color_dict[cls])

        cls_mean = np.array(cls_stats[cls]).mean()
        cls_std = np.array(cls_stats[cls]).std()
        cls_legend = ("Class " + str(cls) + ": {0:.2f}" + "+/-{1:.2f}").format(
            cls_mean, cls_std
        )
        ax[c].legend([cls_legend])
        print(
            ("Class " + str(cls) + ": {0:.2f}" +
             "+/-{1:.2f}").format(cls_mean, cls_std)
        )
        plt.savefig(args.new_path + "/stats/" + "img_{:05d}.png".format(f))
        plt.pause(0.1)
    plt.show()


def read_activity_mnist(name="activity_mnist_small"):
    with open(args.new_path + name + ".pkl", "rb") as f:
        (videos, labels) = pickle.load(f)

    print("Created data: (", videos[0].shape,
          ",", labels[0].shape, ") x ", len(videos))
    print(videos[0][0, :, :].max(), videos[0][0, :, :].min())

    # See some examples
    # view_videos(videos, labels)

    # Get data statistics
    data_stats(videos, labels)

# ------------------------------------------------------------------------------------
if __name__ == "__main__":
    targets = []
    prev_target = None

    # digits = list(range(10))
    # actions = ['horizontal', 'inv-horizontal', 'vertical', 'diagonal', 'inv-diagonal']
    # for i in range(100):
    #     subaction = (random.choice(digits), random.choice(actions), get_color(random.choice(actions)))
    #     if subaction is prev_target:
    #         continue
    #     prev_target = subaction
    #     targets.append(subaction)
    # print(len(targets))

    # check_rootfolders(args.path, "raw")
    # check_rootfolders(args.new_path, "frames-" + args.name)
    create_activity_mnist(
        train=False,
        targets = [
            [(1, "horizontal", get_color('horizontal')),
             (3, "inv-diagonal", get_color('inv-diagonal')),
             (5, "inv-horizontal", get_color('inv-horizontal')),
             (7, "diagonal", get_color('diagonal')),
             (9, "vertical", get_color('vertical')),
             ],
        ],
        file_name=args.name,
    )
    # read_activity_mnist(name=args.name)
