import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from os.path import join
import logging
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils import get_device, set_seeds, create_directory, setup, get_toy_labels
from datasets import BoundingBoxForecastingDataset, future_bounding_box_collate
from datasets.transforms import ImglistToTensor
from networks import ProgressForecastingNet
from networks import RandomNet, StaticNet, RelativeNet
from losses import bo_weight

"""
implementation of https://arxiv.org/abs/1705.01781
"""

def train(network, batch, l1_criterion, l2_criterion, device, optimizer=None):
    video_name, frames, tube, progress_values, future_frames, future_tube, future_progress_values, lengths = batch

    frames = frames.to(device)
    tube = tube.to(device)
    progress_values = progress_values.to(device)

    future_frames.to(device)
    future_tube.to(device)
    future_progress_values.to(device)

    if optimizer:
        optimizer.zero_grad()

    predictions = network(frames, tube, lengths)
    print(predictions)
    exit(0)

    repeated_labels = labels.repeat(network.num_heads, 1, 1)
    l1_loss = l1_criterion(predictions, repeated_labels)
    l2_loss = l2_criterion(predictions, repeated_labels)

    total_l1_loss = torch.sum(l1_loss, dim=-1)
    _, wta_index = torch.min(total_l1_loss, dim=0)
    wta_l1_loss = l1_loss[wta_index[0]]
    wta_predictions = predictions[wta_index[0]].unsqueeze(dim=0)

    # wta_l1_loss, wta_indices = torch.min(l1_loss, dim=0)
    wta_l2_loss = torch.FloatTensor([1, 2])
    # wta_predictions = torch.gather(predictions, 0, wta_indices.unsqueeze(0))
    wta_bo_loss = bo_weight(device, labels, wta_predictions)


    # progress is in range (0, 1], but batch is zero-padded
    # we can use this to multiply our loss with 0s for padded values
    mask = (labels != 0).int().to(device)
    masked_wta_l1_loss = wta_l1_loss * mask
    
    count = lengths.sum()
    if optimizer:
        loss = masked_wta_l1_loss * wta_bo_loss
        loss = loss.sum() / count
        loss.backward()
        optimizer.step()

    return {
        'predictions': predictions,
        'wta_l1_loss': wta_l1_loss,
        'wta_l2_loss': wta_l2_loss,
        'wta_bo_loss': wta_bo_loss,
        'count': count,
    }

def main():
    args, dirs, device = setup()

    # create datasets
    train_set = BoundingBoxForecastingDataset(dirs['dataset_directory'], args.data_type, dirs['annotation_path'], dirs['train_splitfile_path'], 5, transform=ImglistToTensor(dim=0))
    test_set = BoundingBoxForecastingDataset(dirs['dataset_directory'], args.data_type, dirs['annotation_path'], dirs['test_splitfile_path'], 5, transform=ImglistToTensor(dim=0))
    train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True, collate_fn=future_bounding_box_collate)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False, collate_fn=future_bounding_box_collate)

    # load model
    net = ProgressForecastingNet(device, embed_size=args.embed_size, p_dropout=args.dropout_chance).to(device)
    if args.model_name:
        model_path = join(dirs['model_directory'], args.model_name)
        net.load_state_dict(torch.load(model_path))

    # setup network dict for evaluation/testing
    networks = {}
    loss_dict = {'l1_loss': 0.0, 'l2_loss': 0.0, 'count': 0}
    networks['network'] = loss_dict.copy()
    networks['network']['net'] = net
    if args.random:
        networks['random'] = loss_dict.copy()
        networks['random']['net'] = RandomNet(device)
    if args.static:
        networks['static'] = loss_dict.copy()
        networks['static']['net'] = StaticNet(device)
    if args.relative:
        networks['relative'] = loss_dict.copy()
        networks['relative']['net'] = RelativeNet(device, train_set.get_average_tube_frame_length())

    # criterions & optimizer
    l1_criterion = nn.L1Loss(reduction='none')
    l2_criterion = nn.MSELoss(reduction='none')
    optimizer = optim.Adam(net.parameters(), lr=args.learning_rate)

    logging.info(f'[{args.experiment_name}] starting experiment')
    for epoch in range(args.epochs):
        train_bo_loss, train_l1_loss, train_l2_loss, train_count = 0.0, 0.0, 0.0, 0

        if not args.eval:
            net.train()
            for batch in tqdm(train_loader, leave=False):
                batch_result = train(net, batch, l1_criterion, l2_criterion, device, optimizer=optimizer)

                train_l1_loss += batch_result['wta_l1_loss'].sum().item()
                train_l2_loss += batch_result['wta_l2_loss'].sum().item()
                train_count += batch_result['count'].item()

            logging.info(f'[{epoch:03d} train] avg l1 loss {(train_l1_loss / train_count):.4f}, avg l2 loss {(train_l2_loss / train_count):.4f}')
        
            if epoch % args.save_every == 0 and epoch > 0:
                model_name = f'{epoch:03d}.pth'
                model_path = join(dirs['model_directory'], model_name)
                logging.info(f'[{epoch:03d}] saving model {model_name}')
                torch.save(net.state_dict(), model_path)

        net.eval()
        colors = ['r', 'g', 'b', 'c', 'm']
        for batch_index, batch in tqdm(enumerate(test_loader), leave=False, total=len(test_loader)):
            do_figure = args.figures and args.batch_size == 1 and batch_index % args.figure_every == 0
            for model_name in networks:
                batch_result = train(networks[model_name]['net'], batch, l1_criterion, l2_criterion, device)
                networks[model_name]['l1_loss'] += batch_result['wta_l1_loss'].sum().item()
                networks[model_name]['l2_loss'] += batch_result['wta_l2_loss'].sum().item()
                networks[model_name]['count'] += batch_result['count'].item()

                if do_figure:
                    for i,prediction in enumerate(batch_result['predictions']):
                        plot_prediction = prediction.cpu().detach().squeeze().numpy()
                        plt.plot(plot_prediction, label=f'{model_name}_head_{i+1}')

            if do_figure:
                video_names, frames, boxes, labels, lengths = batch
                plot_labels = labels.cpu().detach().squeeze().numpy()
                plt.plot(plot_labels, label='ground truth')
                plt.title('Progress Predictions')
                plt.xlabel('Frame')
                plt.ylabel('Percentag (%)')
                plt.legend(loc='best')
                if 'toy' in args.dataset:
                    action_labels = get_toy_labels(dirs['dataset_directory'], video_names[0])
                    for j, label in enumerate(action_labels):
                        plt.axvspan(j-0.5, j+0.5, facecolor=colors[label], alpha=0.2, zorder=-1)
                plot_name = f'figure_{batch_index}.png'
                plot_path = join(dirs['figures_directory'], plot_name)
                plt.savefig(plot_path)
                plt.clf()

        for model_name in networks:
            logging.info(f'[{epoch:03d} test {model_name}] avg l1 loss {(networks[model_name]["l1_loss"] / networks[model_name]["count"]):.4f}, avg l2 loss {(networks[model_name]["l2_loss"] / networks[model_name]["count"]):.4f}')
        
        if args.eval:
            break # only 1 epoch for evaluation


if __name__ == '__main__':
    main()