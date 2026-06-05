"""Train a linear or MLP multi-label head on saved CheXpert features."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch import nn

from chexpert.data import TARGET_LABELS
from chexpert.engine import (
    apply_standardizer,
    build_loader,
    compute_standardizer,
    load_feature_file,
    print_head_metrics,
    run_head_epoch,
    set_seed,
)
from chexpert.losses import FocalLossWithLogits, compute_focal_alpha, compute_pos_weight
from chexpert.models import build_head


def main():
    parser = argparse.ArgumentParser(
        description="Train a linear or MLP multi-label classifier on saved CheXpert features."
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
    parser.add_argument("--head-type", default="mlp", choices=["linear", "mlp"])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.3)
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
    val_features, val_labels, _ = load_feature_file(args.val_features)

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

    model = build_head(
        head_type=args.head_type,
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        num_labels=num_labels,
        dropout=args.dropout,
    ).to(device)
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
        run_name = f"{args.head_type}_{feature_model}"

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / f"{run_name}_best.pt"

    print(f"Train features: {args.train_features}")
    print(f"Validation features: {args.val_features}")
    print(f"Input dim: {input_dim}")
    print(f"Head type: {args.head_type}")
    if args.head_type == "mlp":
        print(f"Hidden dim: {args.hidden_dim}")
        print(f"Dropout: {args.dropout}")
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
        train_metrics = run_head_epoch(model, train_loader, criterion, device, optimizer)
        val_metrics = run_head_epoch(model, val_loader, criterion, device)

        print(f"\nEpoch {epoch}/{args.epochs}")
        print_head_metrics("  train", train_metrics)
        print_head_metrics("  val", val_metrics)

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
                    "head_type": args.head_type,
                    **(
                        {"hidden_dim": args.hidden_dim, "dropout": args.dropout}
                        if args.head_type == "mlp"
                        else {}
                    ),
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
        test_metrics = run_head_epoch(model, test_loader, criterion, device)
        print_head_metrics("test", test_metrics)


if __name__ == "__main__":
    main()
