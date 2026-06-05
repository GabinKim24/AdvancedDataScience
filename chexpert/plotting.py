"""Matplotlib figures for the evaluation entrypoint.

ROC curves, confusion matrices, class-wise metric bars, and optional t-SNE/UMAP
embeddings of the saved features.
"""

import matplotlib.pyplot as plt
import torch

from chexpert.data import TARGET_LABELS
from chexpert.metrics import compute_roc_curve


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


def plot_finetuned_metric_bars(rows, output_path):
    labels = [row["label"] for row in rows]
    aurocs = [row["auroc"] for row in rows]
    f1s = [row["f1"] for row in rows]
    recalls = [row["recall"] for row in rows]

    x = torch.arange(len(labels)).numpy()
    width = 0.25

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(x - width, aurocs, width, label="AUROC")
    ax.bar(x, f1s, width, label="F1")
    ax.bar(x + width, recalls, width, label="Recall")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_xticks(x, labels=labels, rotation=25, ha="right")
    ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
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
