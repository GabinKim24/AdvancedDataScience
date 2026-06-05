"""Frozen feature extraction for DenseNet121 and DINOv3 backbones."""

from pathlib import Path

import torch

from chexpert.data import TARGET_LABELS
from chexpert.models import pool_dino_features, select_densenet121_features


DEFAULT_SPLITS = [
    "data/processed/train_internal_uzeros.csv",
    "data/processed/intraval_internal_uzeros.csv",
    "data/processed/valid_uzeros.csv",
]


def extract_features(model, dataloader, device, model_type, feature_type, max_batches=None):
    all_features = []
    all_labels = []
    all_paths = []

    with torch.inference_mode():
        for batch_idx, (images, labels, paths) in enumerate(dataloader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            images = images.to(device, non_blocking=True)
            if model_type == "densenet121":
                features = select_densenet121_features(model, images)
            else:
                outputs = model(pixel_values=images)
                features = pool_dino_features(outputs, model.config, feature_type)

            all_features.append(features.cpu())
            all_labels.append(labels.cpu())
            all_paths.extend(list(paths))

            if (batch_idx + 1) % 50 == 0:
                print(f"  processed batches: {batch_idx + 1}")

    return {
        "features": torch.cat(all_features, dim=0),
        "labels": torch.cat(all_labels, dim=0),
        "paths": all_paths,
        "target_labels": TARGET_LABELS,
        "feature_model": (
            "densenet121_imagenet"
            if model_type == "densenet121"
            else model.config.name_or_path
        ),
        "feature_type": None if model_type == "densenet121" else feature_type,
    }


def safe_model_name(model_name):
    return model_name.replace("/", "_").replace("-", "_")


def output_path_for_split(csv_path, output_dir, model_type, model_name, feature_type):
    csv_path = Path(csv_path)
    if model_type == "densenet121":
        return Path(output_dir) / f"{csv_path.stem}_densenet121_features.pt"

    model_tag = safe_model_name(model_name)
    return Path(output_dir) / f"{csv_path.stem}_{model_tag}_{feature_type}_features.pt"
