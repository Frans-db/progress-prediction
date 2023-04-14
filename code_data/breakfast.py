import os
import cv2
from tqdm import tqdm

actions = ['cereals', 'coffee', 'friedegg', 'juice', 'milk', 'pancake', 'salat', 'sandwich', 'scrambledegg', 'tea']
root = '/home/frans/Datasets/breakfast/'

def mkdir_safe(path: str):
    if not os.path.isdir(path):
        os.mkdir(path)

def make_path_safe(root: str, action_class: str, person_id: str, cam: str):
    mkdir_safe(os.path.join(root, action_class))
    mkdir_safe(os.path.join(root, action_class, person_id))
    mkdir_safe(os.path.join(root, action_class, person_id, cam))

    return os.path.join(root, action_class, person_id, cam)

def split_videos():
    video_root = os.path.join(root, 'videos')
    frames_root = os.path.join(root, 'rgb-images')
    mkdir_safe(frames_root)

    video_paths = []
    person_ids = sorted(os.listdir(video_root))
    for person_id in person_ids:
        person_path = os.path.join(video_root, person_id)
        cams = sorted(os.listdir(person_path))
        for cam in cams:
            cam_path = os.path.join(person_path, cam)
            videos = sorted(os.listdir(cam_path))
            for video in videos:
                video_path = os.path.join(cam_path, video)
                if 'labels' in video_path:
                    continue

                for action in actions:
                    if action not in video_path:
                        continue
                    video_paths.append((video_path, (frames_root, action, person_id, cam)))

    for video_path, (frames_root, action, person_id, cam) in tqdm(video_paths):
        save_path = make_path_safe(frames_root, action, person_id, cam)
        vidcap = cv2.VideoCapture(video_path)
        success,image = vidcap.read()
        count = 0
        while success:
            cv2.imwrite(os.path.join(save_path, f'frame_{count:05d}.jpg'), image)     # save frame as JPEG file      
            success,image = vidcap.read()
            count += 1
            
def reorganise_directory(dir_name: str, target_dir: str):
    dir_root = os.path.join(root, dir_name)
    target_root = os.path.join(root, target_dir)
    mkdir_safe(target_root)

    paths = []
    for r, dirs, files in os.walk(dir_root, topdown=False):
        for name in files:
            person_id, cam, _, action_class = name.split('_')
            cam = cam.replace('stereo01', 'stereo')
            action_class = action_class.replace('.txt', '')
            save_path = make_path_safe(target_root, action_class, person_id, cam)
            os.rmdir(save_path)

            paths.append((os.path.join(r, name), f'{save_path}.txt'))
    for (read_path, write_path) in tqdm(paths):
        with open(read_path) as f:
            data = f.read()
        with open(write_path, 'w+') as f:
            f.write(data)



def main():
    # split_videos()
    # reorganise_directory('features', 'new_features')
    # reorganise_directory('groundtruth', 'new_groundtruth')

    create_train_test_splits(list(range(3, 16)), 's1')
    create_train_test_splits(list(range(16, 29)), 's2')
    create_train_test_splits(list(range(29, 42)), 's3')
    create_train_test_splits(list(range(42, 55)), 's4')

def create_train_test_splits(testrange, splitname):
    """
    s1: P03 - P15
    s2: P16 - P28
    s3: P29 - P41
    s4: P42 - P54
    """
    walk_root = '/home/frans/Datasets/breakfast/features/dense_trajectories'
    tests = []
    trains = []
    for root, dirs, files in os.walk(walk_root, topdown=False):
        for name in files:
            if 'txt' not in name:
                continue
            path = os.path.join(root, name)

            contains_action = False
            for action in actions:
                if action in path:
                    contains_action = True
            if not contains_action:
                continue
            
            found = False
            for index in testrange:
                if f'P{index:02d}' in path:
                    tests.append(path.replace(f'{walk_root}/', ''))
                    found = True
                    break

            if not found:
                trains.append(path.replace(f'{walk_root}/', ''))

    print(f'{splitname} train {len(trains)} test {len(tests)} total {len(trains) + len(tests)}')
    with open(f'train_{splitname}.txt', 'w+') as f:
        f.write('\n'.join(trains))
    with open(f'test_{splitname}.txt', 'w+') as f:
        f.write('\n'.join(tests))

if __name__ == '__main__':
    main()