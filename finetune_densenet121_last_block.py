import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import DenseNet121_Weights, densenet121

from chexpert_data import (
    CheXpertFiveLabelDataset,
    TARGET_LABELS,
    get_image_transform,
)
from finetune_dinov3_last_block import (
    compute_pos_weight,
    run_epoch,
    save_summary_csv,
    set_seed,
)


class DenseNet121Classifier(nn.Module):
    def __init__(self, head_type, hidden_dim, dropout, num_labels):
        super().__init__()
        weights = DenseNet121_Weights.DEFAULT
        self.backbone = densenet121(weights=weights)
        feature_dim = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Identity()

        if head_type == "linear":
            self.classifier = nn.Linear(feature_dim, num_labels)
        elif head_type == "mlp":
            self.classifier = nn.Sequential(
                nn.Linear(feature_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_labels),
            )
        else:
            raise ValueError(f"Unknown head_type: {head_type}")

    def forward(self, images):
        features = self.backbone.features(images)
        features = F.relu(features, inplace=False)
        features = F.adaptive_avg_pool2d(features, (1, 1))
        features = torch.flatten(features, 1)
        return self.classifier(features)


def unfreeze_densenet_last_block(model):
    for param in model.backbone.parameters():
        param.requires_grad = False

    trainable_modules = [
        model.backbone.features.denseblock4,
        model.backbone.features.norm5,
    ]
    for module in trainable_modules:
        for param in module.parameters():
            param.requires_grad = True

    for param in model.classifier.parameters():
        param.requires_grad = True

    trainable_backbone = sum(
        param.numel() for param in model.backbone.parameters() if param.requires_grad
    )
    total_backbone = sum(param.numel() for param in model.backbone.parameters())
    classifier_params = sum(param.numel() for param in model.classifier.parameters())
    return trainable_backbone, total_backbone, classifier_params


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune DenseNet121 denseblock4 for CheXpert."
    )
    parser.add_argument("--train-csv", default="data/processed/train_internal_uzeros.csv")
    parser.add_argument("--val-csv", default="data/processed/intraval_internal_uzeros.csv")
    parser.add_argument("--test-csv", default="data/processed/valid_uzeros.csv")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--head-type", default="mlp", choices=["linear", "mlp"])
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--backbone-lr", type=float, default=1e-5)
    parser.add_argument("--head-lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--output-dir", default="checkpoints")
    parser.add_argument(
        "--run-name",
        default="densenet121_denseblock4_mlp_30ep_bs16_w4_h512_drop01",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cuda", "cpu"],
    )
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    args = parser.parse_args()

    set_seed(args.seed)

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
    print("Loading DenseNet121 ImageNet weights...")
    model = DenseNet121Classifier(
        head_type=args.head_type,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        num_labels=len(TARGET_LABELS),
    ).to(device)

    trainable_backbone, total_backbone, classifier_params = unfreeze_densenet_last_block(
        model
    )
    pos_weight = compute_pos_weight(train_dataset).to(device)
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

    print(f"Head type: {args.head_type}")
    print("Unfrozen block path: features.denseblock4 + features.norm5")
    print(f"Trainable backbone params: {trainable_backbone:,} / {total_backbone:,}")
    print(f"Classifier params: {classifier_params:,}")
    print(f"pos_weight: {[round(value, 4) for value in pos_weight.cpu().tolist()]}")

    best_metric = float("-inf")
    best_epoch = 0
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer,
            scaler,
            args.threshold,
            max_batches=args.max_train_batches,
        )
        val_metrics = run_epoch(
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
        print(
            "  train: "
            + ", ".join(f"{key}={value:.4f}" for key, value in train_metrics.items())
        )
        print(
            "  val: "
            + ", ".join(f"{key}={value:.4f}" for key, value in val_metrics.items())
        )

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
                    "model_name": "densenet121_imagenet",
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
    test_metrics = run_epoch(
        model,
        test_loader,
        criterion,
        device,
        None,
        scaler,
        args.threshold,
    )
    print(
        "test: "
        + ", ".join(f"{key}={value:.4f}" for key, value in test_metrics.items())
    )


if __name__ == "__main__":
    main()
