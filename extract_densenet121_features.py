"""Extract frozen DenseNet121 (ImageNet) features for CheXpert CSV splits.

Provenance (see PROVENANCE.md):
  * Frozen-backbone -> cached-feature extraction follows the
    DINOv2ForRadiology linear-probe methodology (DINORAD, CC BY-NC 4.0,
    reimplemented; not a verbatim copy).
  * DenseNet121 + ImageNet weights come from torchvision directly (LIB).
  * The penultimate-feature pooling (features -> ReLU -> global avg pool ->
    flatten) and the saved feature-file schema are the author's own (USED-OWN).
"""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision.models import DenseNet121_Weights, densenet121

from chexpert_data import TARGET_LABELS, build_dataloader


DEFAULT_SPLITS = [
    "data/processed/train_internal_uzeros.csv",
    "data/processed/intraval_internal_uzeros.csv",
    "data/processed/valid_uzeros.csv",
]


def build_densenet121_feature_extractor(device):
    weights = DenseNet121_Weights.DEFAULT
    model = densenet121(weights=weights)
    model.classifier = torch.nn.Identity()
    model.eval()
    model.to(device)
    return model


def extract_features(model, dataloader, device, max_batches=None):
    all_features = []
    all_labels = []
    all_paths = []

    with torch.inference_mode():
        for batch_idx, (images, labels, paths) in enumerate(dataloader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            images = images.to(device, non_blocking=True)

            features = model.features(images)
            features = F.relu(features, inplace=False)
            features = F.adaptive_avg_pool2d(features, (1, 1))
            features = torch.flatten(features, 1)

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
        "feature_model": "densenet121_imagenet",
    }


def output_path_for_split(csv_path, output_dir):
    csv_path = Path(csv_path)
    return Path(output_dir) / f"{csv_path.stem}_densenet121_features.pt"


def main():
    parser = argparse.ArgumentParser(
        description="Extract DenseNet121 ImageNet features for CheXpert CSV splits."
    )
    parser.add_argument("--splits", nargs="+", default=DEFAULT_SPLITS)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--output-dir", default="features")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cuda", "cpu"],
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Optional smoke-test limit. Leave unset for full extraction.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    print(f"Device: {device}")
    print("Loading DenseNet121 ImageNet weights...")
    model = build_densenet121_feature_extractor(device)

    for split_csv in args.splits:
        split_csv = Path(split_csv)
        if not split_csv.exists():
            raise FileNotFoundError(f"CSV not found: {split_csv}")

        print(f"\nExtracting features from: {split_csv}")
        dataloader = build_dataloader(
            csv_path=split_csv,
            data_root=args.data_root,
            image_size=args.image_size,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=False,
        )

        feature_data = extract_features(
            model=model,
            dataloader=dataloader,
            device=device,
            max_batches=args.max_batches,
        )

        output_path = output_path_for_split(split_csv, output_dir)
        torch.save(feature_data, output_path)

        print(f"Saved: {output_path}")
        print(f"  features: {tuple(feature_data['features'].shape)}")
        print(f"  labels: {tuple(feature_data['labels'].shape)}")
        print(f"  paths: {len(feature_data['paths'])}")


if __name__ == "__main__":
    main()
