import torch
import torch.nn as nn
import os
import matplotlib.pyplot as plt

def load_splitfile(path: str):
    with open(path) as f:
        names = f.readlines()
        names = [name.strip() for name in names]
    return names

def get_ucf_data():
    root = '/home/frans/Datasets/ucf24'
    data_root = os.path.join(root, 'rgb-images')
    train_split = 'train.txt'
    test_split = 'test.txt'

    train_names = load_splitfile(os.path.join(root, 'splitfiles', train_split))
    test_names = load_splitfile(os.path.join(root, 'splitfiles', test_split))

    train_lengths, test_lengths = [], []

    for name in train_names:
        video_path = os.path.join(data_root, name)
        frame_names = sorted(os.listdir(video_path))
        train_lengths.append(len(frame_names))
    for name in test_names:
        video_path = os.path.join(data_root, name)
        frame_names = sorted(os.listdir(video_path))
        test_lengths.append(len(frame_names))
            
    return train_lengths, test_lengths

def get_breakfast_data():
    root = '/home/frans/Datasets/breakfast'
    data_root = os.path.join(root, 'features/dense_trajectories')
    train_split = 'train_s1.txt'
    test_split = 'test_s1.txt'

    train_names = load_splitfile(os.path.join(root, 'splitfiles', train_split))
    test_names = load_splitfile(os.path.join(root, 'splitfiles', test_split))
    train_lengths, test_lengths = [], []

    for name in train_names:
        data_path = os.path.join(data_root, f'{name}.txt')
        with open(data_path) as f:
            data = f.readlines()
            train_lengths.append(len(data))
    for name in test_names:
        data_path = os.path.join(data_root, f'{name}.txt')
        with open(data_path) as f:
            data = f.readlines()
            test_lengths.append(len(data))
            
    return train_lengths, test_lengths

def best(train_lengths, test_lengths, name: str):
    average_progress, counts = [], []
    for length in train_lengths:
        for i in range(length):
            if len(average_progress) == i:
                average_progress.append(0)
                counts.append(0)
            average_progress[i] += (i + 1) / length
            counts[i] += 1
    average_progress = [p / c for p,c in zip(average_progress, counts)]
    max_test_length = max(test_lengths)
    for i in range(max_test_length):
        if len(average_progress) == i:
            average_progress.append(1.0)

    criterion = nn.MSELoss(reduction='sum')
    loss, count = 0.0, 0
    for length in test_lengths:
        progress = torch.arange(1, length + 1, 1) / length
        prediction = torch.FloatTensor(average_progress[:length])
        
        loss += criterion(prediction, progress).item()
        count += length
    
    print(loss / count)
    plt.plot(average_progress)
    plt.title(f'Average Progress on {name}')
    plt.xlabel('Frame')
    plt.ylabel('Progress (%)')
    plt.savefig(f'./best_{name}.png')
    plt.clf()

def main():
    print('--- ucf ---')
    train_lengths, test_lengths = get_ucf_data()
    best(train_lengths, test_lengths, 'ucf')
    print('--- breakfast ---')
    train_lengths, test_lengths = get_breakfast_data()
    best(train_lengths, test_lengths, 'bf')



if __name__ == '__main__':
    main()