import torch
import torch.nn as nn
import math

def bo_weight(p, p_hat):
    m = torch.full(p.shape, 0.5).cuda()
    r = torch.full(p.shape, 0.5 * math.sqrt(2)).cuda()

    weight = ((p - m) / r).square() + ((p_hat - m) / r).square()
    return torch.clamp(weight, max=1)