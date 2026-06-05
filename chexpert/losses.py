"""Loss functions and class-imbalance weights for multi-label CheXpert."""

import torch
from torch import nn


def compute_pos_weight(labels):
    """``pos_weight`` for BCE from a label tensor (used by the frozen-head path)."""
    positives = labels.sum(dim=0)
    negatives = labels.shape[0] - positives
    return negatives / positives.clamp_min(1.0)


def compute_pos_weight_from_dataset(dataset):
    """``pos_weight`` from a dataset's ``labels`` array (used by the finetune path)."""
    labels = torch.tensor(dataset.labels, dtype=torch.float32)
    positives = labels.sum(dim=0)
    negatives = labels.shape[0] - positives
    return negatives / positives.clamp_min(1.0)


def compute_focal_alpha(labels, eps=1e-6):
    positives = labels.sum(dim=0)
    negatives = labels.shape[0] - positives
    return (negatives / (positives + negatives).clamp_min(eps)).clamp(0.05, 0.95)


class FocalLossWithLogits(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        bce_loss = nn.functional.binary_cross_entropy_with_logits(
            logits,
            targets,
            reduction="none",
        )
        probs = torch.sigmoid(logits)
        pt = probs * targets + (1 - probs) * (1 - targets)
        focal_factor = (1 - pt).pow(self.gamma)
        loss = focal_factor * bce_loss

        if self.alpha is not None:
            alpha = self.alpha.to(logits.device)
            alpha_factor = alpha * targets + (1 - alpha) * (1 - targets)
            loss = alpha_factor * loss

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss
