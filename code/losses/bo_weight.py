import torch
import torch.nn as nn
import math

def bo_weight(device, p, p_hat):
    m = torch.full(p.shape, 0.5).to(device)
    r = torch.full(p.shape, 0.5 * math.sqrt(2)).to(device)

    weight = ((p - m) / r).square() + ((p_hat - m) / r).square()
    return torch.clamp(weight, max=1)