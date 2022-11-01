import torch
import torch.nn.functional as F


class ReplayLoss:
    def __init__(self):
        self.criterion = F.binary_cross_entropy_with_logits

    def __call__(self, topk_preds, target_nodes, target_values, weights=None):
        mask = (topk_preds.indices == torch.LongTensor(target_nodes).to(topk_preds.indices.device).reshape(-1, 1))
        f = mask.sum(dim=1) > 0

        if weights is not None:
            weights = torch.FloatTensor(weights).to(topk_preds.indices.device)[mask]
        return (
            self.criterion(
                topk_preds.values[mask],
                torch.FloatTensor(target_values).to(topk_preds.indices.device)[f],
                weights
            ),
            topk_preds.values[mask],
            mask
        )


class ReplayLossSlated:
    def __init__(self):
        self.criterion = F.binary_cross_entropy_with_logits

    @staticmethod
    def slated_filter(preds, gt):
        gt = torch.LongTensor(gt).to(preds.device)
        x = gt.repeat_interleave(preds.shape[1], dim=1).reshape(-1, preds.shape[1])
        y = preds.repeat_interleave(preds.shape[1], dim=0)
        c = x == y
        return c.reshape(-1, preds.shape[1], preds.shape[1]).any(1)

    def __call__(self, topk_preds, target_nodes, target_values, weights=None):
        mask = self.slated_filter(topk_preds.indices, target_nodes)
        targets = torch.FloatTensor(target_values).to(topk_preds.indices.device)[mask]
        if weights is not None:
            weights = torch.FloatTensor(weights).to(topk_preds.indices.device)[mask]
        return self.criterion(topk_preds.values[mask], targets, weights), topk_preds.values[mask], mask


def get_ope_loss_function(args):
    if args.slated:
        return ReplayLossSlated()
    return ReplayLoss()