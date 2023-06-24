import torch.nn as nn
import torch

def train_flat_features(network, criterion, batch, max_length, device, optimizer=None, return_results=False):
    l2_loss = nn.MSELoss(reduction="sum")
    l1_loss = nn.L1Loss(reduction="sum")
    _, data, progress = batch
    data = data.to(device)
    B, _ = data.shape
    predicted_progress = network(data).reshape(B)
    if return_results:
        return predicted_progress.cpu()
    progress = progress.to(device)
    if optimizer:
        optimizer.zero_grad()
        criterion(predicted_progress, progress).backward()
        optimizer.step()

    return {
        "l2_loss": l2_loss(predicted_progress, progress).item(),
        "l1_loss": l1_loss(predicted_progress * 100, progress * 100).item(),
        "count": B,
    }


def train_flat_frames(network, criterion, batch, max_length, device, optimizer=None, return_results=False):
    l2_loss = nn.MSELoss(reduction="sum")
    l1_loss = nn.L1Loss(reduction="sum")
    progress = batch[-1]
    data = batch[1:-1]
    data = tuple([d.to(device) for d in data])

    B = data[0].shape[0]
    predicted_progress = network(*data).reshape(B)
    if return_results:
        return predicted_progress.cpu()
    progress = progress.to(device)
    if optimizer:
        optimizer.zero_grad()
        criterion(predicted_progress, progress).backward()
        optimizer.step()

    return {
        "l2_loss": l2_loss(predicted_progress, progress).item(),
        "l1_loss": l1_loss(predicted_progress * 100, progress * 100).item(),
        "count": B,
    }


def train_progress(network, criterion, batch, max_length, device, optimizer=None, return_results=False):
    l2_loss = nn.MSELoss(reduction="sum")
    l1_loss = nn.L1Loss(reduction="sum")
    progress = batch[-1]

    # TODO: Split batch into segments of max_length
    data = batch[1:-1]
    S = data[0].shape[1]

    predicted_progress = []
    data = tuple([torch.split(d, max_length, dim=1) for d in data])
    if getattr(network, 'reset', None):
        network.reset()
    for samples in zip(*data):
        samples = tuple([sample.to(device) for sample in samples])
        predicted_progress.append(network(*samples))
    predicted_progress = torch.cat(predicted_progress, dim=-1)

    if return_results:
        return predicted_progress.cpu()
    progress = progress.to(device)
    if optimizer:
        optimizer.zero_grad()
        criterion(predicted_progress, progress).backward()
        optimizer.step()

    return {
        "l2_loss": l2_loss(predicted_progress, progress).item(),
        "l1_loss": l1_loss(predicted_progress * 100, progress * 100).item(),
        "count": S,
    }


def train_rsd(network, criterion, batch, max_length, device, optimizer=None, return_results=False):
    l2_loss = nn.MSELoss(reduction="sum")
    l1_loss = nn.L1Loss(reduction="sum")
    smooth_l1_loss = nn.SmoothL1Loss(reduction="sum")

    rsd = batch[-2] / network.rsd_normalizer
    progress = batch[-1]
    S = progress.shape[1]

    data = batch[1:-2]
    data = tuple([d.to(device) for d in data])

    predicted_rsd, predicted_progress = network(*data)
    if return_results:
        return predicted_progress.cpu()
    rsd = rsd.to(device)
    progress = progress.to(device)
    if optimizer:
        optimizer.zero_grad()
        loss = criterion(predicted_rsd, rsd) + criterion(predicted_progress, progress)
        loss.backward()
        optimizer.step()

    return {
        "rsd_l1_loss": l1_loss(predicted_rsd, rsd),
        "rsd_smooth_l1_loss": smooth_l1_loss(predicted_rsd, rsd),
        "rsd_l2_loss": l2_loss(predicted_rsd, rsd),
        "rsd_normal_l1_loss": l1_loss(
            predicted_rsd * network.rsd_normalizer, rsd * network.rsd_normalizer
        ),
        "l1_loss": l1_loss(predicted_progress * 100, progress * 100),
        "smooth_l1_loss": smooth_l1_loss(predicted_progress, progress),
        "l2_loss": l2_loss(predicted_progress, progress),
        "count": S,
    }


def embed_frames(network, batch, device, batch_size: int):
    data = batch[1:-1]
    data = tuple([torch.split(d.squeeze(dim=0), batch_size) for d in data])

    embeddings = []
    for samples in zip(*data):
        samples = tuple([sample.to(device) for sample in samples])
        sample_embeddings = network.embed(*samples)
        embeddings.extend(sample_embeddings.tolist())

    return batch[0][0], embeddings