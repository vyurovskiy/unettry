# from torch import nn
# from torch.nn import functional as F
# import torch


def dice_flat(pred, target):
    m1 = pred.view(-1)
    m2 = target.view(-1)
    return 2. * (m1 * m2).mean() / (m1.mean() + m2.mean() + 1e-7)


def dice(pred, target):
    n = pred.shape[0]
    m1 = pred.view(n, -1)
    m2 = target.view(n, -1)
    return (2. * (m1 * m2).mean(1) / (m1.mean(1) + m2.mean(1) + 1e-7)).mean()


def dice_loss(probs, targets):
    return 1 - dice(probs, targets)
