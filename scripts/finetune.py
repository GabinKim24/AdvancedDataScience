"""Fine-tune the last blocks of DenseNet121 or DINOv3 for CheXpert."""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch import nn

from chexpert.data import CheXpertFiveLabelDataset, TARGET_LABELS, get_image_transform
from chexpert.engine import (
    print_finetune_metrics,
    run_finetune_epoch,
    save_summary_csv,
    set_seed,
)
from chexpert.losses import compute_pos_weight_from_dataset
from chexpert.models import (
    DenseNet121Classifier,
    DINOv3Classifier,
    load_dinov3_backbone,
    unfreeze_densenet_last_block,
    unfreeze_last_blocks,
)


DEFAULT_MODEL_NAME = "facebook/dinov3-vitb16-pretrain-lvd1689m"


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune DenseNet121 or DINOv3 for CheXpert."
    )
    parser.add_argument("--model", default="dinov3", choices=["densenet121", "dinov3"])
    parser.add_argument("--train-csv", default="data/processed/train_internal_uzeros.csv")
    parser.add_argument("--val-csv", default="data/processed/intraval_internal_uzeros.csv")
    parser.add_argument("--test-csv", default="data/processed/valid_uzeros.csv")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument(
        "--feature-type",
        default="mean_patch",
        choices=["pooled", "cls", "mean_patch", "max_patch", "mean_max_patch"],
    )
    parser.add_argument("--head-type", default="mlp", choices=["linear", "mlp"])
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--backbone-lr", type=float, default=1e-5)
    parser.add_argument("--head-lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--unfreeze-last-blocks", type=int, default=1)
    parser.add_argument("--output-dir", default="checkpoints")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cuda", "cpu"],
    )
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"))
    parser.add_argument(
        "--max-train-batches",
        type=int,
        default=None,
        help="Optional smoke-test limit. Use a small value to estimate runtime.",
    )
    parser.add_argument("--max-val-batches", type=int, default=None)
    args = parser.parse_args()

    set_seed(args.seed)

    if args.batch_size is None:
        args.batch_size = 16 if args.model == "densenet121" else 8
    if args.num_workers is None:
        args.num_workers = 2 if args.model == "densenet121" else 0
    if args.epochs is None:
        args.epochs = 30 if args.model == "densenet121" else 5
    if args.patience is None:
        args.patience = 5 if args.model == "densenet121" else 2
    if args.run_name is None:
        args.run_name = (
            "densenet121_denseblock4_mlp"
            if args.model == "densenet121"
            else "dinov3_last_block_mean_patch_mlp"
        )

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")

    transform = get_image_transform(args.image_size)
    train_dataset = CheXpertFiveLabelDataset(args.train_csv, args.data_root, transform)
    val_dataset = CheXpertFiveLabelDataset(args.val_csv, args.data_root, transform)
    test_dataset = CheXpertFiveLabelDataset(args.test_csv, args.data_root, transform)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    if args.model == "densenet121":
        print("Loading DenseNet121 ImageNet weights...")
        model = DenseNet121Classifier(
            head_type=args.head_type,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
            num_labels=len(TARGET_LABELS),
        ).to(device)
        block_path, trainable_backbone, total_backbone = unfreeze_densenet_last_block(
            model
        )
        checkpoint_model_name = "densenet121_imagenet"
        checkpoint_feature_type = None
    else:
        print(f"Loading DINOv3 model: {args.model_name}")
        backbone = load_dinov3_backbone(args.model_name, args.hf_token)
        block_path, trainable_backbone, total_backbone = unfreeze_last_blocks(
            backbone, args.unfreeze_last_blocks
        )
        model = DINOv3Classifier(
            backbone=backbone,
            feature_type=args.feature_type,
            head_type=args.head_type,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
            num_labels=len(TARGET_LABELS),
        ).to(device)
        checkpoint_model_name = args.model_name
        checkpoint_feature_type = args.feature_type

    pos_weight = compute_pos_weight_from_dataset(train_dataset).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        [
            {
                "params": [p for p in model.backbone.parameters() if p.requires_grad],
                "lr": args.backbone_lr,
            },
            {"params": model.classifier.parameters(), "lr": args.head_lr},
        ],
        weight_decay=args.weight_decay,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / f"{args.run_name}_best.pt"
    metrics_path = output_dir / f"{args.run_name}_metrics.csv"

    classifier_params = sum(param.numel() for param in model.classifier.parameters())
    if args.model == "dinov3":
        print(f"Feature type: {args.feature_type}")
    print(f"Head type: {args.head_type}")
    print(f"Unfrozen block path: {block_path}")
    print(f"Trainable backbone params: {trainable_backbone:,} / {total_backbone:,}")
    print(f"Classifier params: {classifier_params:,}")
    print(f"pos_weight: {[round(value, 4) for value in pos_weight.cpu().tolist()]}")

    best_metric = float("-inf")
    best_epoch = 0
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_finetune_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer,
            scaler,
            args.threshold,
            max_batches=args.max_train_batches,
        )
        val_metrics = run_finetune_epoch(
            model,
            val_loader,
            criterion,
            device,
            None,
            scaler,
            args.threshold,
            max_batches=args.max_val_batches,
        )

        print(f"\nEpoch {epoch}/{args.epochs}")
        print_finetune_metrics("  train", train_metrics)
        print_finetune_metrics("  val", val_metrics)

        row = {"epoch": epoch}
        row.update({f"train_{key}": value for key, value in train_metrics.items()})
        row.update({f"val_{key}": value for key, value in val_metrics.items()})
        history.append(row)
        save_summary_csv(metrics_path, history)

        current_metric = val_metrics["macro_auroc"]
        if current_metric > best_metric:
            best_metric = current_metric
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_name": checkpoint_model_name,
                    "feature_type": checkpoint_feature_type,
                    "head_type": args.head_type,
                    "hidden_dim": args.hidden_dim,
                    "dropout": args.dropout,
                    "target_labels": TARGET_LABELS,
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

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics = run_finetune_epoch(
        model, test_loader, criterion, device, None, scaler, args.threshold
    )
    print_finetune_metrics("test", test_metrics)

    summary_path = output_dir / f"{args.run_name}_summary.json"
    summary = {
        "run_name": args.run_name,
        "checkpoint": str(checkpoint_path),
        "best_epoch": best_epoch,
        "best_val_metric": best_metric,
        "test_metrics": test_metrics,
        "args": vars(args),
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
