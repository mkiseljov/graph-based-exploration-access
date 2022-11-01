import numpy as np
import torch


def ndcg_score(mask, targets, inverse_weights):
    d = 1 / np.log2(np.arange(mask.shape[1]) + 2).reshape(1, -1)
    return (((targets.reshape(mask.shape[0], -1) * mask) * d) * inverse_weights.reshape(mask.shape[0], -1)).sum()  # sum cuz we want to calculate cumulative rolling ndcg


def ndcg_score_replayed(mask, targets, inverse_weights):
    targets = targets.reshape(mask.shape[0], -1)
    inverse_weights = inverse_weights.reshape(mask.shape[0], -1)
    tmp = mask * np.arange(mask.shape[1]).reshape(1, -1)
    tmp += 2 + (-tmp.min(1).reshape(-1, 1)) * mask
    d = mask * (1 / np.log2(tmp))
    return (targets * mask * d * inverse_weights).sum()


def ap_score(mask, targets, inverse_weights):
    targets = targets.reshape(mask.shape[0], -1)
    inverse_weights = inverse_weights.reshape(mask.shape[0], -1)
    recall = (mask * targets) / ((mask * targets).sum(1).reshape(-1, 1) + 1e-20)
    precision = (mask * targets) / (np.cumsum(mask, axis=1) + 1e-20)
    return (precision * recall * inverse_weights).sum()
