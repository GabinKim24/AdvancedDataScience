import argparse
import random
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from chexpert_data import TARGET_LABELS


def load_feature_file(path):
    try:
        data = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        data = torch.load(path, map_location="cpu")

    features = data["features"].float()
    labels = data["labels"].float()
    return features, labels, data


def compute_standardizer(features, eps=1e-6):
    mean = features.mean(dim=0, keepdim=True)
    std = features.std(dim=0, keepdim=True, unbiased=False).clamp_min(eps)
    return mean, std


def apply_standardizer(features, mean, std):
    return (features - mean) / std


def compute_pos_weight(labels):
    positives = labels.sum(dim=0)
    negatives = labels.shape[0] - positives
    return negatives / positives.clamp_min(1.0)


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_loader(features, labels, batch_size, shuffle, pin_memory):
    dataset = TensorDataset(features, labels)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=pin_memory,
    )


class LinearHead(nn.Module):
    def __init__(self, input_dim, num_labels):
        super().__init__()
        self.classifier = nn.Linear(input_dim, num_labels)

    def forward(self, features):
        return self.classifier(features)


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


def compute_focal_alpha(labels, eps=1e-6):
    positives = labels.sum(dim=0)
    negatives = labels.shape[0] - positives
    return (negatives / (positives + negatives).clamp_min(eps)).clamp(0.05, 0.95)


def safe_divide(numerator, denominator):
    if denominator == 0:
        return 0.0
    return numerator / denominator


def binary_metrics_from_logits(logits, labels, threshold=0.5):
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
            safe_divide(2 * precision[i].item() * recall[i].item(), precision[i].item() + recall[i].item())
            for i in range(labels.shape[1])
        ]
    )

    return {
        "macro_precision": precision.mean().item(),
        "macro_recall": recall.mean().item(),
        "macro_f1": f1.mean().item(),
    }


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


def auroc_metrics(logits, labels):
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


def run_epoch(model, loader, criterion, device, optimizer=None):
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

    metrics = {"loss": total_loss / total_examples}
    metrics.update(binary_metrics_from_logits(logits, labels))
    metrics.update(auroc_metrics(logits, labels))
    return metrics


def print_metrics(prefix, metrics):
    keys = ["loss", "macro_auroc", "macro_f1", "macro_precision", "macro_recall"]
    parts = []
    for key in keys:
        if key in metrics:
            parts.append(f"{key}={metrics[key]:.4f}")
    print(f"{prefix}: " + ", ".join(parts))


def main():
    parser = argparse.ArgumentParser(
        description="Train a linear multi-label classifier on saved CheXpert features."
    )
    parser.add_argument(
        "--train-features",
        default="features/train_internal_uzeros_densenet121_features.pt",
    )
    parser.add_argument(
        "--val-features",
        default="features/intraval_internal_uzeros_densenet121_features.pt",
    )
    parser.add_argument("--test-features", default=None)
    parser.add_argument("--output-dir", default="checkpoints")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--loss", default="bce", choices=["bce", "focal"])
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--no-focal-alpha", action="store_true")
    parser.add_argument("--no-standardize", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cuda", "cpu"],
    )
    args = parser.parse_args()

    set_seed(args.seed)

    train_features, train_labels, train_meta = load_feature_file(args.train_features)
    val_features, val_labels, val_meta = load_feature_file(args.val_features)

    input_dim = train_features.shape[1]
    num_labels = train_labels.shape[1]
    if val_features.shape[1] != input_dim:
        raise ValueError("Train and validation feature dimensions do not match.")
    if num_labels != len(TARGET_LABELS):
        raise ValueError(f"Expected {len(TARGET_LABELS)} labels, got {num_labels}.")

    standardize = not args.no_standardize
    if standardize:
        feature_mean, feature_std = compute_standardizer(train_features)
        train_features = apply_standardizer(train_features, feature_mean, feature_std)
        val_features = apply_standardizer(val_features, feature_mean, feature_std)
    else:
        feature_mean = None
        feature_std = None

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")

    pin_memory = device.type == "cuda"
    train_loader = build_loader(
        train_features,
        train_labels,
        args.batch_size,
        shuffle=True,
        pin_memory=pin_memory,
    )
    val_loader = build_loader(
        val_features,
        val_labels,
        args.batch_size,
        shuffle=False,
        pin_memory=pin_memory,
    )

    model = LinearHead(input_dim=input_dim, num_labels=num_labels).to(device)
    pos_weight = compute_pos_weight(train_labels).to(device)
    focal_alpha = compute_focal_alpha(train_labels).to(device)
    if args.loss == "bce":
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        criterion = FocalLossWithLogits(
            alpha=None if args.no_focal_alpha else focal_alpha,
            gamma=args.focal_gamma,
        )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    run_name = args.run_name
    if run_name is None:
        feature_model = str(train_meta.get("feature_model", "features")).replace("/", "_")
        run_name = f"linear_{feature_model}"

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / f"{run_name}_best.pt"

    print(f"Train features: {args.train_features}")
    print(f"Validation features: {args.val_features}")
    print(f"Input dim: {input_dim}")
    print(f"Output labels: {TARGET_LABELS}")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Loss: {args.loss}")
    if args.loss == "focal":
        print(f"focal_gamma: {args.focal_gamma}")
        print(
            "focal_alpha: "
            f"{None if args.no_focal_alpha else [round(value, 4) for value in focal_alpha.cpu().tolist()]}"
        )
    print(f"Standardize features: {standardize}")
    print(f"pos_weight: {[round(value, 4) for value in pos_weight.cpu().tolist()]}")

    best_metric = float("-inf")
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, criterion, device, optimizer)
        val_metrics = run_epoch(model, val_loader, criterion, device)

        print(f"\nEpoch {epoch}/{args.epochs}")
        print_metrics("  train", train_metrics)
        print_metrics("  val", val_metrics)

        current_metric = val_metrics.get("macro_auroc", -val_metrics["loss"])
        if current_metric > best_metric:
            best_metric = current_metric
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "input_dim": input_dim,
                    "num_labels": num_labels,
                    "target_labels": TARGET_LABELS,
                    "feature_model": train_meta.get("feature_model"),
                    "feature_type": train_meta.get("feature_type"),
                    "standardize": standardize,
                    "feature_mean": feature_mean,
                    "feature_std": feature_std,
                    "pos_weight": pos_weight.cpu(),
                    "loss": args.loss,
                    "focal_gamma": args.focal_gamma,
                    "focal_alpha": None if args.no_focal_alpha else focal_alpha.cpu(),
                    "best_epoch": best_epoch,
                    "best_val_metric": best_metric,
                    "args": vars(args),
                },
                checkpoint_path,
            )
            print(f"  saved best checkpoint: {checkpoint_path}")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"Early stopping at epoch {epoch}. Best epoch: {best_epoch}")
                break

    print(f"\nBest checkpoint: {checkpoint_path}")
    print(f"Best epoch: {best_epoch}")

    if args.test_features:
        test_features, test_labels, _ = load_feature_file(args.test_features)
        if standardize:
            test_features = apply_standardizer(test_features, feature_mean, feature_std)
        test_loader = build_loader(
            test_features,
            test_labels,
            args.batch_size,
            shuffle=False,
            pin_memory=pin_memory,
        )

        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        test_metrics = run_epoch(model, test_loader, criterion, device)
        print_metrics("test", test_metrics)


if __name__ == "__main__":
    main()
