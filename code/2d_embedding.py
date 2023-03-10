import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader
from os.path import join
import logging
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

from utils import setup
from datasets import ImageDataset
from datasets.transforms import ImglistToTensor
from networks import RSDNet

"""
implementation of https://arxiv.org/abs/1705.01781
"""

def train(network, batch, smooth_l1_criterion, l1_criterion, l2_criterion, device, optimizer=None):
    frames, labels = batch
    frames = frames.to(device)
    labels = labels.float().to(device)

    if optimizer:
        optimizer.zero_grad()
    predictions = network(frames)
    loss = smooth_l1_criterion(predictions.squeeze(), labels)
    l1_loss = l1_criterion(predictions.squeeze(), labels)
    l2_loss = l2_criterion(predictions.squeeze(), labels)

    if optimizer:
        avg_loss = loss.sum() / frames.shape[0]
        avg_loss.backward()
        optimizer.step()

    return predictions, loss, l1_loss, l2_loss, frames.shape[0]

def main():
    args, dirs, device = setup()
    logging.info(f'[{args.experiment_name}] starting experiment')

    # create datasets
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224))
    ])
    test_set = ImageDataset(dirs['dataset_directory'], args.data_type, dirs['test_splitfile_path'], transform=transform)

    # load model
    net = models.resnet18().to(device)
    net.fc = nn.Sequential(
        nn.Linear(512, 1),
        nn.Sigmoid()
    ).to(device)

    if args.model_name:
        model_path = join(dirs['model_directory'], args.model_name)
        logging.info(f'[{args.experiment_name}] loading model {model_path}')
        net.load_state_dict(torch.load(model_path))

    net.fc = nn.Identity()

    # criterions & optimizer
    smooth_l1_criterion = nn.SmoothL1Loss(reduction='none')
    l1_criterion = nn.L1Loss(reduction='none')
    l2_criterion = nn.MSELoss(reduction='none')
    
    video_names = sorted(os.listdir(os.path.join(dirs['dataset_directory'], args.data_type)))
    embedding_directory = os.path.join(dirs['dataset_directory'], args.embedding_directory)
    os.mkdir(embedding_directory)

    for video_name in tqdm(video_names):
        video_embedding_path = os.path.join(embedding_directory, f'{video_name}.txt')

        test_set.frame_paths, test_set.progress = test_set._get_data([video_name])
        test_loader = DataLoader(test_set, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)

        activations = []
        for batch_index, batch in enumerate(test_loader):
            frames, labels = batch
            frames = frames.to(device)
            predictions = net(frames)

            activations.append(predictions)
        concatenated = torch.cat(activations)
        concatenated = concatenated.detach().cpu().tolist()
        text_rows = []
        for row in concatenated:
            text_rows.append(' '.join(map(str, row)))
        with open(video_embedding_path, 'w+') as f:
            f.write('\n'.join(text_rows))



if __name__ == '__main__':
    main()