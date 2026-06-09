import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from chexpert_data import TARGET_LABELS


class LinearHead(nn.Module):
    def __init__(self, input_dim, num_labels):
        super().__init__()
        self.classifier = nn.Linear(input_dim, num_labels)

    def forward(self, features):
        return self.classifier(features)


class MLPHead(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_labels, dropout):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_labels),
        )

    def forward(self, features):
        return self.classifier(features)


def torch_load(path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def load_features(path):
    data = torch_load(path, map_location="cpu")
    return data["features"].float(), data["labels"].float(), data


def build_model(checkpoint):
    input_dim = checkpoint["input_dim"]
    num_labels = checkpoint["num_labels"]

    if "hidden_dim" in checkpoint:
        return MLPHead(
            input_dim=input_dim,
            hidden_dim=checkpoint["hidden_dim"],
            num_labels=num_labels,
            dropout=checkpoint.get("dropout", 0.0),
        )

    return LinearHead(input_dim=input_dim, num_labels=num_labels)


def standardize_if_needed(features, checkpoint):
    if checkpoint.get("standardize", False):
        return (features - checkpoint["feature_mean"]) / checkpoint["feature_std"]
    return features


def predict_logits(model, features, batch_size, device):
    loader = DataLoader(TensorDataset(features), batch_size=batch_size, shuffle=False)
    logits = []

    model.eval()
    with torch.inference_mode():
        for (batch_features,) in loader:
            batch_features = batch_features.to(device)
            logits.append(model(batch_features).cpu())

    return torch.cat(logits, dim=0)


def binary_confusion_counts(y_true, y_pred):
    tp = int(((y_true == 1) & (y_pred == 1)).sum().item())
    tn = int(((y_true == 0) & (y_pred == 0)).sum().item())
    fp = int(((y_true == 0) & (y_pred == 1)).sum().item())
    fn = int(((y_true == 1) & (y_pred == 0)).sum().item())
    return tn, fp, fn, tp


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
    ranks = torch.arange(1, sorted_scores.numel() + 1, dtype=torch.float32)
    _, counts = torch.unique_consecutive(sorted_scores, return_counts=True)

    average_ranks = torch.empty_like(ranks)
    start = 0
    for count in counts.tolist():
        end = start + count
        average_ranks[start:end] = ranks[start:end].mean()
        start = end

    positive_rank_sum = average_ranks[sorted_targets == 1].sum().item()
    return (positive_rank_sum - positives * (positives + 1) / 2) / (
        positives * negatives
    )


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


def compute_metrics(logits, labels, threshold, pos_weight=None):
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


def write_csv(path, rows):
    if not rows:
        return

    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_confusion_matrix(row, output_path):
    matrix = [
        [row["tn"], row["fp"]],
        [row["fn"], row["tp"]],
    ]

    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_title(row["label"])
    ax.set_xticks([0, 1], labels=["Pred 0", "Pred 1"])
    ax.set_yticks([0, 1], labels=["True 0", "True 1"])

    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(matrix[i][j]), ha="center", va="center", color="black")

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_roc_curves(probs, labels, rows, output_path):
    fig, ax = plt.subplots(figsize=(6, 5))

    for idx, row in enumerate(rows):
        fpr, tpr, _ = compute_roc_curve(probs[:, idx], labels[:, idx])
        if not fpr:
            continue
        ax.plot(fpr, tpr, label=f"{row['label']} ({row['auroc']:.3f})")

    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_metric_bars(rows, output_path):
    labels = [row["label"] for row in rows]
    metrics = ["auroc", "f1", "precision", "recall"]
    x = torch.arange(len(labels)).float()
    width = 0.18

    fig, ax = plt.subplots(figsize=(10, 5))
    for idx, metric in enumerate(metrics):
        values = [row[metric] for row in rows]
        ax.bar((x + (idx - 1.5) * width).tolist(), values, width=width, label=metric)

    ax.set_xticks(x.tolist(), labels=labels, rotation=20, ha="right")
    ax.set_ylim(0, 1)
    ax.set_title("Class-wise Performance")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def choose_visualization_labels(labels):
    positive_counts = labels.sum(dim=1)
    primary = labels.argmax(dim=1)
    primary[positive_counts == 0] = -1
    return primary


def sample_for_embedding(features, labels, max_samples, seed):
    n = features.shape[0]
    if n <= max_samples:
        return features, labels

    generator = torch.Generator().manual_seed(seed)
    idx = torch.randperm(n, generator=generator)[:max_samples]
    return features[idx], labels[idx]


def plot_tsne(features, labels, output_path, max_samples, seed):
    try:
        from sklearn.manifold import TSNE
    except ImportError:
        print("Skipping t-SNE plot because scikit-learn is not installed.")
        return

    sample_features, sample_labels = sample_for_embedding(
        features,
        labels,
        max_samples=max_samples,
        seed=seed,
    )
    y = choose_visualization_labels(sample_labels)

    embedding = TSNE(
        n_components=2,
        perplexity=30,
        init="pca",
        learning_rate="auto",
        random_state=seed,
    ).fit_transform(sample_features.numpy())

    plot_embedding(embedding, y, "t-SNE Feature Embedding", output_path)


def plot_umap(features, labels, output_path, max_samples, seed):
    try:
        import umap
    except ImportError:
        print("Skipping UMAP plot because umap-learn is not installed.")
        return

    sample_features, sample_labels = sample_for_embedding(
        features,
        labels,
        max_samples=max_samples,
        seed=seed,
    )
    y = choose_visualization_labels(sample_labels)

    embedding = umap.UMAP(
        n_components=2,
        n_neighbors=15,
        min_dist=0.1,
        random_state=seed,
    ).fit_transform(sample_features.numpy())

    plot_embedding(embedding, y, "UMAP Feature Embedding", output_path)


def plot_embedding(embedding, y, title, output_path):
    fig, ax = plt.subplots(figsize=(7, 6))
    names = ["No positive"] + TARGET_LABELS

    for value in [-1, 0, 1, 2, 3, 4]:
        mask = y.numpy() == value
        if not mask.any():
            continue
        label = names[value + 1]
        ax.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            s=6,
            alpha=0.6,
            label=label,
        )

    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(markerscale=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def evaluate_one_model(args):
    checkpoint = torch_load(args.checkpoint, map_location="cpu")
    features, labels, feature_meta = load_features(args.features)
    features = standardize_if_needed(features, checkpoint)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")

    model = build_model(checkpoint).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    logits = predict_logits(model, features, args.batch_size, device)
    pos_weight = checkpoint.get("pos_weight")
    metrics, rows, probs, preds = compute_metrics(
        logits=logits,
        labels=labels,
        threshold=args.threshold,
        pos_weight=pos_weight,
    )

    run_name = args.run_name or Path(args.checkpoint).stem
    output_dir = Path(args.output_dir) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "run_name": run_name,
        "checkpoint": args.checkpoint,
        "features": args.features,
        "feature_model": feature_meta.get("feature_model"),
        "feature_type": feature_meta.get("feature_type"),
        "best_epoch": checkpoint.get("best_epoch"),
        "best_val_metric": checkpoint.get("best_val_metric"),
        **metrics,
    }

    with (output_dir / "summary_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    write_csv(output_dir / "summary_metrics.csv", [summary])
    write_csv(output_dir / "classwise_metrics.csv", rows)

    torch.save(
        {
            "logits": logits,
            "probs": probs,
            "preds": preds,
            "labels": labels,
            "target_labels": TARGET_LABELS,
            "threshold": args.threshold,
        },
        output_dir / "predictions.pt",
    )

    plot_metric_bars(rows, output_dir / "classwise_metrics.png")
    plot_roc_curves(probs, labels, rows, output_dir / "roc_curves.png")

    confusion_dir = output_dir / "confusion_matrices"
    confusion_dir.mkdir(exist_ok=True)
    for row in rows:
        plot_confusion_matrix(
            row,
            confusion_dir / f"{row['label'].replace(' ', '_')}_confusion_matrix.png",
        )

    if args.save_embeddings:
        plot_tsne(
            features,
            labels,
            output_dir / "tsne_features.png",
            max_samples=args.max_embedding_samples,
            seed=args.seed,
        )
        plot_umap(
            features,
            labels,
            output_dir / "umap_features.png",
            max_samples=args.max_embedding_samples,
            seed=args.seed,
        )

    print(f"Saved evaluation outputs to: {output_dir}")
    print(
        "Summary: "
        f"macro_auroc={summary['macro_auroc']:.4f}, "
        f"macro_f1={summary['macro_f1']:.4f}, "
        f"weighted_bce_loss={summary.get('weighted_bce_loss', float('nan')):.4f}, "
        f"unweighted_bce_loss={summary['unweighted_bce_loss']:.4f}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate trained Linear/MLP heads and save metrics/plots."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--features", required=True)
    parser.add_argument("--output-dir", default="evaluation")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--save-embeddings", action="store_true")
    parser.add_argument("--max-embedding-samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cuda", "cpu"],
    )
    args = parser.parse_args()

    evaluate_one_model(args)


if __name__ == "__main__":
    main()
