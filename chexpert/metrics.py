"""Multi-label classification metrics.

Two metric "styles" co-exist because the project's experiments were tuned with
them and their numbers must stay reproducible:

* the frozen-head path (:mod:`scripts.train_head`) prefers
  :func:`sklearn.metrics.roc_auc_score` (via :func:`auroc_metrics`) and reports
  macro precision/recall/F1 as a dict (:func:`macro_binary_metrics`);
* the fine-tune / evaluation paths use the dependency-free
  :func:`torch_binary_auroc` directly.

They are kept as distinct, explicitly named functions rather than merged so the
exact numeric behavior of each path is preserved.
"""

import torch
from torch import nn

from chexpert.data import TARGET_LABELS


def safe_divide(numerator, denominator):
    return 0.0 if denominator == 0 else numerator / denominator


def torch_binary_auroc(scores, targets):
    targets = targets.float()
    positives = targets.sum().item()
    negatives = targets.numel() - positives

    if positives == 0 or negatives == 0:
        return float("nan")

    sorted_scores, order = torch.sort(scores.float())
    sorted_targets = targets[order]

    ranks = torch.arange(
        1,
        sorted_scores.numel() + 1,
        dtype=torch.float32,
        device=sorted_scores.device,
    )
    _, counts = torch.unique_consecutive(sorted_scores, return_counts=True)

    start = 0
    average_ranks = torch.empty_like(ranks)
    for count in counts.tolist():
        end = start + count
        average_ranks[start:end] = ranks[start:end].mean()
        start = end

    positive_rank_sum = average_ranks[sorted_targets == 1].sum().item()
    auc = (positive_rank_sum - positives * (positives + 1) / 2) / (
        positives * negatives
    )
    return auc


def macro_binary_metrics(logits, labels, threshold=0.5):
    """Macro precision/recall/F1 as a dict (frozen-head training path)."""
    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).float()

    tp = (preds * labels).sum(dim=0)
    fp = (preds * (1 - labels)).sum(dim=0)
    fn = ((1 - preds) * labels).sum(dim=0)

    precision = torch.tensor(
        [safe_divide(tp[i].item(), (tp[i] + fp[i]).item()) for i in range(labels.shape[1])]
    )
    recall = torch.tensor(
        [safe_divide(tp[i].item(), (tp[i] + fn[i]).item()) for i in range(labels.shape[1])]
    )
    f1 = torch.tensor(
        [
            safe_divide(
                2 * precision[i].item() * recall[i].item(),
                precision[i].item() + recall[i].item(),
            )
            for i in range(labels.shape[1])
        ]
    )

    return {
        "macro_precision": precision.mean().item(),
        "macro_recall": recall.mean().item(),
        "macro_f1": f1.mean().item(),
    }


def precision_recall_f1(logits, labels, threshold=0.5):
    """Per-label precision/recall/F1 tensors (fine-tune path)."""
    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).float()

    tp = (preds * labels).sum(dim=0)
    fp = (preds * (1 - labels)).sum(dim=0)
    fn = ((1 - preds) * labels).sum(dim=0)

    precision = torch.tensor(
        [safe_divide(tp[i].item(), (tp[i] + fp[i]).item()) for i in range(labels.shape[1])]
    )
    recall = torch.tensor(
        [safe_divide(tp[i].item(), (tp[i] + fn[i]).item()) for i in range(labels.shape[1])]
    )
    f1 = torch.tensor(
        [
            safe_divide(
                2 * precision[i].item() * recall[i].item(),
                precision[i].item() + recall[i].item(),
            )
            for i in range(labels.shape[1])
        ]
    )
    return precision, recall, f1


def auroc_metrics(logits, labels):
    """Per-label + macro AUROC, preferring scikit-learn when available."""
    try:
        from sklearn.metrics import roc_auc_score
    except ImportError:
        roc_auc_score = None

    probs_tensor = torch.sigmoid(logits).cpu()
    labels_tensor = labels.cpu()
    probs = probs_tensor.numpy()
    y_true = labels_tensor.numpy()

    per_label = {}
    valid_scores = []
    for idx, label_name in enumerate(TARGET_LABELS):
        if len(set(y_true[:, idx].tolist())) < 2:
            continue
        if roc_auc_score is None:
            score = torch_binary_auroc(probs_tensor[:, idx], labels_tensor[:, idx])
        else:
            score = roc_auc_score(y_true[:, idx], probs[:, idx])
        per_label[f"auroc_{label_name}"] = float(score)
        valid_scores.append(float(score))

    if not valid_scores:
        return per_label

    per_label["macro_auroc"] = sum(valid_scores) / len(valid_scores)
    return per_label


def compute_finetune_metrics(logits, labels, loss, threshold):
    """Summary metrics dict for the fine-tune training loop."""
    probs = torch.sigmoid(logits)
    precision, recall, f1 = precision_recall_f1(logits, labels, threshold)

    aurocs = []
    for idx in range(labels.shape[1]):
        aurocs.append(torch_binary_auroc(probs[:, idx], labels[:, idx]))

    return {
        "loss": loss,
        "macro_auroc": sum(aurocs) / len(aurocs),
        "macro_f1": f1.mean().item(),
        "macro_precision": precision.mean().item(),
        "macro_recall": recall.mean().item(),
    }


def binary_confusion_counts(y_true, y_pred):
    tp = int(((y_true == 1) & (y_pred == 1)).sum().item())
    tn = int(((y_true == 0) & (y_pred == 0)).sum().item())
    fp = int(((y_true == 0) & (y_pred == 1)).sum().item())
    fn = int(((y_true == 1) & (y_pred == 0)).sum().item())
    return tn, fp, fn, tp


def compute_roc_curve(scores, targets):
    try:
        from sklearn.metrics import roc_curve

        fpr, tpr, thresholds = roc_curve(targets.numpy(), scores.numpy())
        return fpr.tolist(), tpr.tolist(), thresholds.tolist()
    except ImportError:
        order = torch.argsort(scores, descending=True)
        sorted_targets = targets[order].float()
        positives = sorted_targets.sum().item()
        negatives = sorted_targets.numel() - positives

        if positives == 0 or negatives == 0:
            return [], [], []

        tps = torch.cumsum(sorted_targets, dim=0)
        fps = torch.cumsum(1 - sorted_targets, dim=0)
        tpr = torch.cat([torch.tensor([0.0]), tps / positives, torch.tensor([1.0])])
        fpr = torch.cat([torch.tensor([0.0]), fps / negatives, torch.tensor([1.0])])
        return fpr.tolist(), tpr.tolist(), []


def compute_eval_metrics(logits, labels, threshold, pos_weight=None):
    """Full per-class evaluation report used by the evaluation entrypoint."""
    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).float()

    unweighted_loss = nn.BCEWithLogitsLoss()(logits, labels).item()
    metrics = {
        "threshold": threshold,
        "unweighted_bce_loss": unweighted_loss,
    }

    if pos_weight is not None:
        weighted_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)(logits, labels).item()
        metrics["weighted_bce_loss"] = weighted_loss

    rows = []
    aurocs = []
    precisions = []
    recalls = []
    f1s = []

    for idx, label_name in enumerate(TARGET_LABELS):
        y_true = labels[:, idx]
        y_pred = preds[:, idx]
        y_score = probs[:, idx]

        tn, fp, fn, tp = binary_confusion_counts(y_true, y_pred)
        precision = safe_divide(tp, tp + fp)
        recall = safe_divide(tp, tp + fn)
        f1 = safe_divide(2 * precision * recall, precision + recall)
        specificity = safe_divide(tn, tn + fp)
        accuracy = safe_divide(tp + tn, tp + tn + fp + fn)
        auroc = torch_binary_auroc(y_score, y_true)

        rows.append(
            {
                "label": label_name,
                "auroc": auroc,
                "f1": f1,
                "precision": precision,
                "recall": recall,
                "specificity": specificity,
                "accuracy": accuracy,
                "tp": tp,
                "tn": tn,
                "fp": fp,
                "fn": fn,
                "positive_count": int(y_true.sum().item()),
                "negative_count": int((1 - y_true).sum().item()),
            }
        )

        aurocs.append(auroc)
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    metrics.update(
        {
            "macro_auroc": sum(aurocs) / len(aurocs),
            "macro_f1": sum(f1s) / len(f1s),
            "macro_precision": sum(precisions) / len(precisions),
            "macro_recall": sum(recalls) / len(recalls),
        }
    )

    return metrics, rows, probs, preds
