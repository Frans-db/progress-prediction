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

def train(network, batch, l1_criterion, l2_criterion, device, future_weight, reconstruction_weight, optimizer=None):
    video_name, frames, tube, progress_values, future_frames, future_tube, future_progress_values, lengths = batch

    frames = frames.to(device)
    tube = tube.to(device)
    progress_values = progress_values.to(device)

    # future_frames.to(device)
    # future_tube.to(device)
    future_progress_values = future_progress_values.to(device)

    if optimizer:
        optimizer.zero_grad()

    embeddings, progress_predictions, forecasted_embeddings, forecasted_progress_predictions = network(frames, tube, lengths)

    # TODO: Removed mask for now, testing only with batch size = 1
    # # progress is in range (0, 1], but batch is zero-padded
    # # we can use this to multiply our loss with 0s for padded values
    # mask = (progress_values != 0).int().to(device)
    # progress_predictions = progress_predictions * mask

    gt_embeddings = torch.zeros_like(embeddings).to(device)
    gt_embeddings[:, :-network.delta_t, :] = embeddings[:, network.delta_t:, :]

    bo_progress_loss = bo_weight(device, progress_values, progress_predictions)
    future_bo_progress_loss = bo_weight(device, future_progress_values, forecasted_progress_predictions)
    l1_progress_loss = l1_criterion(progress_predictions, progress_values)
    l2_progress_loss = l2_criterion(progress_predictions, progress_values)
    future_l1_progress_loss = l1_criterion(forecasted_progress_predictions, future_progress_values)
    future_l2_progress_loss = l2_criterion(forecasted_progress_predictions, future_progress_values)
    l2_reconstruction_loss = l2_criterion(forecasted_embeddings, embeddings)
    count = lengths.sum()

    if optimizer:
        progress_loss = l1_progress_loss * bo_progress_loss
        progress_loss = progress_loss.sum() / count
        future_progress_loss = future_l1_progress_loss * future_bo_progress_loss
        future_progress_loss = future_progress_loss.sum() / count
        reconstruction_loss = l2_reconstruction_loss.sum() / count

        loss = progress_loss + future_weight * future_progress_loss + reconstruction_weight * reconstruction_loss
        loss.backward()
        optimizer.step()

    return {
        'predictions': progress_predictions,
        'forecasted_predictions': forecasted_progress_predictions,
        'l1_progress_loss': l1_progress_loss,
        'l2_progress_loss': l2_progress_loss,
        'bo_progress_loss': bo_progress_loss,
        'l2_reconstruction_loss': l2_reconstruction_loss,
        'count': count
    }


def main():
    args, dirs, device = setup()

    # create datasets
    train_set = BoundingBoxForecastingDataset(dirs['dataset_directory'], args.data_type, dirs['annotation_path'], dirs['train_splitfile_path'], args.delta_t, transform=ImglistToTensor(dim=0))
    test_set = BoundingBoxForecastingDataset(dirs['dataset_directory'], args.data_type, dirs['annotation_path'], dirs['test_splitfile_path'], args.delta_t, transform=ImglistToTensor(dim=0))
    train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True, collate_fn=future_bounding_box_collate)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False, collate_fn=future_bounding_box_collate)

    # load model
    net = ProgressForecastingNet(device, embed_size=args.embed_size, p_dropout=args.dropout_chance, delta_t=args.delta_t).to(device)
    if args.model_name:
        model_path = join(dirs['model_directory'], args.model_name)
        net.load_state_dict(torch.load(model_path))

    # setup network dict for evaluation/testing
    networks = {}
    loss_dict = {'l1_progress_loss': 0.0, 'l2_progress_loss': 0.0, 'l2_reconstruction_loss': 0.0, 'count': 0}
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
        train_bo_loss, train_l1_loss, train_l2_loss, train_reconstruction_loss, train_count = 0.0, 0.0, 0.0, 0.0, 0

        if not args.eval:
            net.train()
            for batch in tqdm(train_loader, leave=False):
                batch_result = train(net, batch, l1_criterion, l2_criterion, device, args.future_weight, args.reconstruction_weight, optimizer=optimizer)

                train_l1_loss += batch_result['l1_progress_loss'].sum().item()
                train_l2_loss += batch_result['l2_progress_loss'].sum().item()
                train_reconstruction_loss += batch_result['l2_reconstruction_loss'].sum().item()
                train_count += batch_result['count'].item()

            logging.info(f'[{epoch:03d} train] avg l1 loss {(train_l1_loss / train_count):.4f}, avg l2 loss {(train_l2_loss / train_count):.4f}, avg reconstruction loss {(train_reconstruction_loss / train_count):.4f}')
        
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
                batch_result = train(networks[model_name]['net'], batch, l1_criterion, l2_criterion, device, args.future_weight, args.reconstruction_weight)
                networks[model_name]['l1_progress_loss'] += batch_result['l1_progress_loss'].sum().item()
                networks[model_name]['l2_progress_loss'] += batch_result['l2_progress_loss'].sum().item()
                networks[model_name]['l2_reconstruction_loss'] += batch_result['l2_reconstruction_loss'].sum().item()
                networks[model_name]['count'] += batch_result['count'].item()

                if do_figure:
                    predictions = batch_result['predictions'].cpu().detach().squeeze().tolist()
                    forecasted_predictions = batch_result['forecasted_predictions'].cpu().detach().squeeze().tolist()

                    xs = [i for i,_ in enumerate(predictions)]
                    plt.plot(xs, predictions, label='predicted progress')
                    plt.plot([x+net.delta_t for x in xs], forecasted_predictions, label='forecasted progress')

            if do_figure:
                video_names, frames, tube, progress_values, future_frames, future_tube, future_progress_values, lengths = batch
                plot_labels = progress_values.cpu().detach().squeeze().numpy()
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
            logging.info(f'[{epoch:03d} test {model_name}] avg l1 loss {(networks[model_name]["l1_progress_loss"] / networks[model_name]["count"]):.4f}, avg l2 loss {(networks[model_name]["l2_progress_loss"] / networks[model_name]["count"]):.4f}, avg reconstruction loss {(networks[model_name]["l2_reconstruction_loss"] / networks[model_name]["count"]):.4f}')
        
        if args.eval:
            break # only 1 epoch for evaluation


if __name__ == '__main__':
    main()