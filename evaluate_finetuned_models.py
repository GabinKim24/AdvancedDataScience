import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from chexpert_data import (
    CheXpertFiveLabelDataset,
    TARGET_LABELS,
    get_image_transform,
)
from evaluate_heads import (
    compute_metrics,
    plot_confusion_matrix,
    plot_roc_curves,
    write_csv,
)
from finetune_densenet121_last_block import DenseNet121Classifier
from finetune_dinov3_last_block import DINOv3Classifier, compute_pos_weight


def torch_load(path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def load_dinov3_backbone(model_name, hf_token):
    try:
        from transformers import AutoModel
    except ImportError as exc:
        raise ImportError("Install transformers first: pip install transformers") from exc

    return AutoModel.from_pretrained(model_name, token=hf_token)


def build_model_from_checkpoint(checkpoint, device, hf_token):
    model_name = checkpoint.get("model_name", "")
    args = checkpoint.get("args", {})
    head_type = checkpoint.get("head_type", args.get("head_type", "mlp"))
    hidden_dim = checkpoint.get("hidden_dim", args.get("hidden_dim", 512))
    dropout = checkpoint.get("dropout", args.get("dropout", 0.0))

    if "dinov3" in model_name:
        backbone = load_dinov3_backbone(model_name, hf_token)
        model = DINOv3Classifier(
            backbone=backbone,
            feature_type=checkpoint.get("feature_type", args.get("feature_type")),
            head_type=head_type,
            hidden_dim=hidden_dim,
            dropout=dropout,
            num_labels=len(TARGET_LABELS),
        )
    elif model_name == "densenet121_imagenet":
        model = DenseNet121Classifier(
            head_type=head_type,
            hidden_dim=hidden_dim,
            dropout=dropout,
            num_labels=len(TARGET_LABELS),
        )
    else:
        raise ValueError(f"Unsupported fine-tuned checkpoint model_name: {model_name}")

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def predict(model, loader, device):
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


def save_summary_csv(path, summary):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        import csv

        writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)


def plot_classwise_metrics(rows, output_path):
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


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate fine-tuned DINOv3/DenseNet121 image checkpoints."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--eval-csv", default="data/processed/valid_uzeros.csv")
    parser.add_argument("--train-csv", default="data/processed/train_internal_uzeros.csv")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--output-dir", default="evaluation/finetuned")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cuda", "cpu"],
    )
    parser.add_argument("--hf-token", default=None)
    args = parser.parse_args()

    device = torch.device(args.device)
    checkpoint_path = Path(args.checkpoint)
    checkpoint = torch_load(checkpoint_path, map_location=device)
    run_name = args.run_name or checkpoint_path.stem.replace("_best", "")

    transform = get_image_transform(args.image_size)
    eval_dataset = CheXpertFiveLabelDataset(args.eval_csv, args.data_root, transform)
    train_dataset = CheXpertFiveLabelDataset(args.train_csv, args.data_root, transform)
    loader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    print(f"Device: {device}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Eval CSV: {args.eval_csv}")
    model = build_model_from_checkpoint(checkpoint, device, args.hf_token)

    logits, labels, paths = predict(model, loader, device)
    pos_weight = compute_pos_weight(train_dataset)
    metrics, rows, probs, preds = compute_metrics(
        logits, labels, args.threshold, pos_weight=pos_weight
    )

    output_dir = Path(args.output_dir) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "run_name": run_name,
        "checkpoint": str(checkpoint_path),
        "eval_csv": args.eval_csv,
        "model_name": checkpoint.get("model_name", ""),
        "feature_type": checkpoint.get("feature_type", ""),
        "head_type": checkpoint.get("head_type", checkpoint.get("args", {}).get("head_type", "")),
        "best_epoch": checkpoint.get("best_epoch", ""),
        "best_val_metric": checkpoint.get("best_val_metric", ""),
        **metrics,
    }

    save_summary_csv(output_dir / "summary_metrics.csv", summary)
    with (output_dir / "summary_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    write_csv(output_dir / "classwise_metrics.csv", rows)
    torch.save(
        {"logits": logits, "probs": probs, "preds": preds, "labels": labels, "paths": paths},
        output_dir / "predictions.pt",
    )

    plot_classwise_metrics(rows, output_dir / "classwise_metrics.png")
    plot_roc_curves(probs, labels, rows, output_dir / "roc_curves.png")
    confusion_dir = output_dir / "confusion_matrices"
    confusion_dir.mkdir(exist_ok=True)
    for row in rows:
        filename = row["label"].replace(" ", "_") + "_confusion_matrix.png"
        plot_confusion_matrix(row, confusion_dir / filename)

    print(json.dumps(summary, indent=2))
    print(f"Saved evaluation to: {output_dir}")


if __name__ == "__main__":
    main()
