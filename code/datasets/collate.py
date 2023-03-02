import torch
from torch.nn.utils.rnn import pad_sequence

def bounding_box_collate(batch):
    video_names, frames, boxes, labels = zip(*batch)
    lengths = torch.Tensor([sample.shape[0] for sample in frames])

    padded_frames = pad_sequence(frames, batch_first=True)
    padded_boxes = pad_sequence(boxes, batch_first=True)
    padded_labels = pad_sequence(labels, batch_first=True)

    return video_names, padded_frames, padded_boxes, padded_labels, lengths