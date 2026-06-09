"""Fine-tune the last transformer block(s) of DINOv3 for CheXpert.

Provenance (see PROVENANCE.md): this is the end-to-end fine-tuning path of
MohammedSB/DINOv2ForRadiology (DINORAD, CC BY-NC 4.0) — freeze the backbone,
unfreeze the last block(s), and train the backbone + head together with a low
backbone learning rate and a higher head learning rate (cf. their `--fine-tune`
/ `--backbone-learning-rate` linear-probe path). Reimplemented for a standalone
HF Transformers pipeline; the DINOv3 backbone is loaded via `AutoModel` (LIB).
The token pooling, AMP loop, pos_weight BCE, and checkpoint schema are the
author's own (USED-OWN).

Defaults match the project's final DINOv3 fine-tune run
(`dinov3_last_block_mean_patch_mlp_30ep_bs16_w4_h512_drop01`).
"""

import argparse
import csv
import json
import os
import random
from pathlib import Path

import torch
from torch import nn

from chexpert_data import (
    CheXpertFiveLabelDataset,
    TARGET_LABELS,
    get_image_transform,
)


DEFAULT_MODEL_NAME = "facebook/dinov3-vitb16-pretrain-lvd1689m"


class DINOv3Classifier(nn.Module):
    def __init__(self, backbone, feature_type, head_type, hidden_dim, dropout, num_labels):
        super().__init__()
        self.backbone = backbone
        self.feature_type = feature_type

        hidden_size = backbone.config.hidden_size
        input_dim = hidden_size * 2 if feature_type == "mean_max_patch" else hidden_size

        if head_type == "linear":
            self.classifier = nn.Linear(input_dim, num_labels)
        elif head_type == "mlp":
            self.classifier = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_labels),
            )
        else:
            raise ValueError(f"Unknown head_type: {head_type}")

    def select_features(self, outputs):
        if self.feature_type == "pooled":
            if getattr(outputs, "pooler_output", None) is None:
                raise ValueError("This model output does not contain pooler_output.")
            return outputs.pooler_output

        hidden_states = outputs.last_hidden_state
        if self.feature_type == "cls":
            return hidden_states[:, 0, :]

        num_register_tokens = getattr(self.backbone.config, "num_register_tokens", 0)
        patch_start = 1 + num_register_tokens
        patch_tokens = hidden_states[:, patch_start:, :]

        if self.feature_type == "mean_patch":
            return patch_tokens.mean(dim=1)
        if self.feature_type == "max_patch":
            return patch_tokens.max(dim=1).values
        if self.feature_type == "mean_max_patch":
            mean_patch = patch_tokens.mean(dim=1)
            max_patch = patch_tokens.max(dim=1).values
            return torch.cat([mean_patch, max_patch], dim=1)

        raise ValueError(f"Unknown feature_type: {self.feature_type}")

    def forward(self, pixel_values):
        outputs = self.backbone(pixel_values=pixel_values)
        features = self.select_features(outputs)
        return self.classifier(features)


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def torch_load_model(model_name, hf_token):
    try:
        from transformers import AutoModel
    except ImportError as exc:
        raise ImportError("Install transformers first: pip install transformers") from exc

    return AutoModel.from_pretrained(model_name, token=hf_token)


def get_nested_attr(obj, path):
    current = obj
    for name in path.split("."):
        if not hasattr(current, name):
            return None
        current = getattr(current, name)
    return current


def unfreeze_last_blocks(backbone, num_blocks):
    for param in backbone.parameters():
        param.requires_grad = False

    block_paths = [
        "encoder.layer",
        "encoder.layers",
        "encoder.blocks",
        "layer",
        "layers",
        "blocks",
    ]

    blocks = None
    used_path = None
    for path in block_paths:
        candidate = get_nested_attr(backbone, path)
        if candidate is not None and hasattr(candidate, "__len__") and len(candidate) > 0:
            blocks = candidate
            used_path = path
            break

    if blocks is None:
        raise RuntimeError(
            "Could not find transformer blocks. Inspect the model with "
            "`for name, _ in model.named_modules(): print(name)`."
        )

    if num_blocks < 1:
        raise ValueError("--unfreeze-last-blocks must be >= 1.")

    selected_blocks = list(blocks)[-num_blocks:]
    for block in selected_blocks:
        for param in block.parameters():
            param.requires_grad = True

    for norm_name in ["layernorm", "layer_norm", "final_layer_norm", "norm"]:
        module = getattr(backbone, norm_name, None)
        if module is not None:
            for param in module.parameters():
                param.requires_grad = True

    trainable = sum(param.numel() for param in backbone.parameters() if param.requires_grad)
    total = sum(param.numel() for param in backbone.parameters())
    return used_path, trainable, total


def compute_pos_weight(dataset):
    labels = torch.tensor(dataset.labels, dtype=torch.float32)
    positives = labels.sum(dim=0)
    negatives = labels.shape[0] - positives
    return negatives / positives.clamp_min(1.0)


def safe_divide(numerator, denominator):
    return 0.0 if denominator == 0 else numerator / denominator


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
            safe_divide(
                2 * precision[i].item() * recall[i].item(),
                precision[i].item() + recall[i].item(),
            )
            for i in range(labels.shape[1])
        ]
    )
    return precision, recall, f1


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


def compute_metrics(logits, labels, loss, threshold):
    probs = torch.sigmoid(logits)
    precision, recall, f1 = binary_metrics_from_logits(logits, labels, threshold)

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


def run_epoch(
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
    return compute_metrics(
        logits=logits,
        labels=labels,
        loss=total_loss / total_examples,
        threshold=threshold,
    )


def print_metrics(prefix, metrics):
    print(
        f"{prefix}: "
        f"loss={metrics['loss']:.4f}, "
        f"macro_auroc={metrics['macro_auroc']:.4f}, "
        f"macro_f1={metrics['macro_f1']:.4f}, "
        f"macro_precision={metrics['macro_precision']:.4f}, "
        f"macro_recall={metrics['macro_recall']:.4f}"
    )


def save_summary_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune the last transformer block of DINOv3 for CheXpert."
    )
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
    parser.add_argument("--unfreeze-last-blocks", type=int, default=1)
    parser.add_argument("--output-dir", default="checkpoints")
    parser.add_argument(
        "--run-name",
        default="dinov3_last_block_mean_patch_mlp_30ep_bs16_w4_h512_drop01",
    )
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
    print(f"Loading DINOv3 model: {args.model_name}")
    backbone = torch_load_model(args.model_name, args.hf_token)

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

    classifier_params = sum(param.numel() for param in model.classifier.parameters())
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
        print_metrics("  train", train_metrics)
        print_metrics("  val", val_metrics)

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
                    "model_name": args.model_name,
                    "feature_type": args.feature_type,
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
        model, test_loader, criterion, device, None, scaler, args.threshold
    )
    print_metrics("test", test_metrics)

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
