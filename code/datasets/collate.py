import torch
from torch.nn.utils.rnn import pad_sequence


def bounding_box_collate(batch):
    video_names, frames, boxes, labels = zip(*batch)
    lengths = torch.Tensor([sample.shape[0] for sample in frames])

    padded_frames = pad_sequence(frames, batch_first=True)
    padded_boxes = pad_sequence(boxes, batch_first=True)
    padded_labels = pad_sequence(labels, batch_first=True)

    return video_names, padded_frames, padded_boxes, padded_labels, lengths


def progress_collate(batch):
    video_names, frames, labels = zip(*batch)
    lengths = torch.Tensor([sample.shape[0] for sample in frames])

    padded_frames = pad_sequence(frames, batch_first=True)
    padded_labels = pad_sequence(labels, batch_first=True)

    return video_names, padded_frames, padded_labels, lengths


def rsd_collate(batch):
    video_names, frames, rsd_labels, progress_labels = zip(*batch)
    lengths = torch.Tensor([sample.shape[0] for sample in frames])

    padded_frames = pad_sequence(frames, batch_first=True)
    padded_rsd_labels = pad_sequence(rsd_labels, batch_first=True)
    padded_progress_labels = pad_sequence(progress_labels, batch_first=True)

    return video_names, padded_frames, padded_rsd_labels, padded_progress_labels, lengths
