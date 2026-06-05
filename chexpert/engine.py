"""Training/evaluation loops and shared run utilities.

Two epoch loops mirror the torchxrayvision ``train_epoch`` / ``valid_test_epoch``
split, kept separate because the frozen-head path trains on cached feature
tensors while the fine-tune path trains end-to-end on images with AMP.
"""

import csv
import random

import torch
from torch.utils.data import DataLoader, TensorDataset

from chexpert import metrics


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------- #
# Checkpoint / feature-file I/O
# --------------------------------------------------------------------------- #
def torch_load(path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def load_feature_file(path):
    data = torch_load(path, map_location="cpu")
    features = data["features"].float()
    labels = data["labels"].float()
    return features, labels, data


# --------------------------------------------------------------------------- #
# Feature standardization
# --------------------------------------------------------------------------- #
def compute_standardizer(features, eps=1e-6):
    mean = features.mean(dim=0, keepdim=True)
    std = features.std(dim=0, keepdim=True, unbiased=False).clamp_min(eps)
    return mean, std


def apply_standardizer(features, mean, std):
    return (features - mean) / std


def standardize_if_needed(features, checkpoint):
    if checkpoint.get("standardize", False):
        return (features - checkpoint["feature_mean"]) / checkpoint["feature_std"]
    return features


# --------------------------------------------------------------------------- #
# Loaders
# --------------------------------------------------------------------------- #
def build_loader(features, labels, batch_size, shuffle, pin_memory):
    dataset = TensorDataset(features, labels)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=pin_memory,
    )


# --------------------------------------------------------------------------- #
# Epoch loops
# --------------------------------------------------------------------------- #
def run_head_epoch(model, loader, criterion, device, optimizer=None):
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_examples = 0
    all_logits = []
    all_labels = []

    for features, labels in loader:
        features = features.to(device)
        labels = labels.to(device)

        logits = model(features)
        loss = criterion(logits, labels)

        if is_train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        batch_size = features.shape[0]
        total_loss += loss.item() * batch_size
        total_examples += batch_size
        all_logits.append(logits.detach().cpu())
        all_labels.append(labels.detach().cpu())

    logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_labels, dim=0)

    epoch_metrics = {"loss": total_loss / total_examples}
    epoch_metrics.update(metrics.macro_binary_metrics(logits, labels))
    epoch_metrics.update(metrics.auroc_metrics(logits, labels))
    return epoch_metrics


def run_finetune_epoch(
    model,
    loader,
    criterion,
    device,
    optimizer,
    scaler,
    threshold,
    max_batches=None,
):
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_examples = 0
    all_logits = []
    all_labels = []

    for batch_idx, (images, labels, _) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                logits = model(images)
                loss = criterion(logits, labels)

            if is_train:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        batch_size = labels.shape[0]
        total_loss += loss.detach().item() * batch_size
        total_examples += batch_size
        all_logits.append(logits.detach().cpu())
        all_labels.append(labels.detach().cpu())

    logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_labels, dim=0)
    return metrics.compute_finetune_metrics(
        logits=logits,
        labels=labels,
        loss=total_loss / total_examples,
        threshold=threshold,
    )


# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #
def predict_logits(model, features, batch_size, device):
    loader = DataLoader(TensorDataset(features), batch_size=batch_size, shuffle=False)
    logits = []

    model.eval()
    with torch.inference_mode():
        for (batch_features,) in loader:
            batch_features = batch_features.to(device)
            logits.append(model(batch_features).cpu())

    return torch.cat(logits, dim=0)


def predict_image_logits(model, loader, device):
    logits = []
    labels = []
    paths = []

    with torch.inference_mode():
        for images, batch_labels, batch_paths in loader:
            images = images.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                batch_logits = model(images)
            logits.append(batch_logits.cpu())
            labels.append(batch_labels.cpu())
            paths.extend(list(batch_paths))

    return torch.cat(logits, dim=0), torch.cat(labels, dim=0), paths


# --------------------------------------------------------------------------- #
# Console + CSV reporting
# --------------------------------------------------------------------------- #
def print_head_metrics(prefix, epoch_metrics):
    keys = ["loss", "macro_auroc", "macro_f1", "macro_precision", "macro_recall"]
    parts = []
    for key in keys:
        if key in epoch_metrics:
            parts.append(f"{key}={epoch_metrics[key]:.4f}")
    print(f"{prefix}: " + ", ".join(parts))


def print_finetune_metrics(prefix, epoch_metrics):
    print(
        f"{prefix}: "
        f"loss={epoch_metrics['loss']:.4f}, "
        f"macro_auroc={epoch_metrics['macro_auroc']:.4f}, "
        f"macro_f1={epoch_metrics['macro_f1']:.4f}, "
        f"macro_precision={epoch_metrics['macro_precision']:.4f}, "
        f"macro_recall={epoch_metrics['macro_recall']:.4f}"
    )


def write_csv(path, rows):
    if not rows:
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_summary_csv(path, summary):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)


def save_summary_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
