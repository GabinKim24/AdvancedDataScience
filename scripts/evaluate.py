"""Evaluate frozen feature-head or fine-tuned image checkpoints."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from chexpert.data import CheXpertFiveLabelDataset, TARGET_LABELS, get_image_transform
from chexpert.engine import (
    load_feature_file,
    predict_image_logits,
    predict_logits,
    standardize_if_needed,
    torch_load,
    write_csv,
    write_summary_csv,
)
from chexpert.losses import compute_pos_weight_from_dataset
from chexpert.metrics import compute_eval_metrics
from chexpert.models import build_finetuned_model, build_head_model
from chexpert.plotting import (
    plot_confusion_matrix,
    plot_finetuned_metric_bars,
    plot_metric_bars,
    plot_roc_curves,
    plot_tsne,
    plot_umap,
)


def evaluate_one_model(args):
    checkpoint = torch_load(args.checkpoint, map_location="cpu")
    features, labels, feature_meta = load_feature_file(args.features)
    features = standardize_if_needed(features, checkpoint)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")

    model = build_head_model(checkpoint).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    logits = predict_logits(model, features, args.batch_size, device)
    pos_weight = checkpoint.get("pos_weight")
    metrics, rows, probs, preds = compute_eval_metrics(
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

    plot_finetuned_metric_bars(rows, output_dir / "classwise_metrics.png")
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


def evaluate_finetuned_model(args):
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
    model = build_finetuned_model(checkpoint, device, args.hf_token)

    logits, labels, paths = predict_image_logits(model, loader, device)
    pos_weight = compute_pos_weight_from_dataset(train_dataset)
    metrics, rows, probs, preds = compute_eval_metrics(
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
        "best_epoch": checkpoint.get("best_epoch"),
        "best_val_metric": checkpoint.get("best_val_metric"),
        **metrics,
    }

    write_summary_csv(output_dir / "summary_metrics.csv", summary)
    with (output_dir / "summary_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    write_csv(output_dir / "classwise_metrics.csv", rows)

    torch.save(
        {
            "logits": logits,
            "probs": probs,
            "preds": preds,
            "labels": labels,
            "paths": paths,
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

    print(f"Saved evaluation to: {output_dir}")
    print(
        "Summary: "
        f"macro_auroc={summary['macro_auroc']:.4f}, "
        f"macro_f1={summary['macro_f1']:.4f}, "
        f"macro_precision={summary['macro_precision']:.4f}, "
        f"macro_recall={summary['macro_recall']:.4f}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate frozen feature-head or fine-tuned image checkpoints."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--features", default=None)
    parser.add_argument("--output-dir", default="evaluation")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--save-embeddings", action="store_true")
    parser.add_argument("--max-embedding-samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-csv", default="data/processed/valid_uzeros.csv")
    parser.add_argument("--train-csv", default="data/processed/train_internal_uzeros.csv")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--hf-token", default=None)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cuda", "cpu"],
    )
    args = parser.parse_args()

    if args.features:
        if args.batch_size is None:
            args.batch_size = 8192
        evaluate_one_model(args)
    else:
        if args.batch_size is None:
            args.batch_size = 32
        if args.output_dir == "evaluation":
            args.output_dir = "evaluation/finetuned"
        evaluate_finetuned_model(args)


if __name__ == "__main__":
    main()
