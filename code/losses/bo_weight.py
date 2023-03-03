import torch
import torch.nn as nn

def bo_weight(p, p_hat):
    m = torch.full(p.shape, 0.5).cuda()
    r = torch.full(p.shape, 0.5 * torch.sqrt(2)).cuda()

    weight = ((p - m) / r).square() + ((p_hat - m) / r).square()
    return torch.clamp(weight, max=1)