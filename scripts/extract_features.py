"""Extract frozen DenseNet121 or DINOv3 features for CheXpert CSV splits."""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from chexpert.data import build_dataloader
from chexpert.features import DEFAULT_SPLITS, extract_features, output_path_for_split
from chexpert.models import (
    build_densenet121_feature_extractor,
    build_dinov3_feature_extractor,
)


def main():
    parser = argparse.ArgumentParser(
        description="Extract DenseNet121 or DINOv3 features for CheXpert CSV splits."
    )
    parser.add_argument("--model", default="dinov3", choices=["densenet121", "dinov3"])
    parser.add_argument("--splits", nargs="+", default=DEFAULT_SPLITS)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--output-dir", default="features")
    parser.add_argument(
        "--model-name",
        default="facebook/dinov3-vits16-pretrain-lvd1689m",
        help="Hugging Face DINOv3 model id.",
    )
    parser.add_argument(
        "--feature-type",
        default="pooled",
        choices=["pooled", "cls", "mean_patch", "max_patch", "mean_max_patch"],
        help="Global feature representation to save.",
    )
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cuda", "cpu"],
    )
    parser.add_argument(
        "--hf-token",
        default=os.environ.get("HF_TOKEN"),
        help="Hugging Face token. Defaults to HF_TOKEN environment variable.",
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
    batch_size = args.batch_size
    if batch_size is None:
        batch_size = 32 if args.model == "densenet121" else 16

    if args.model == "densenet121":
        print("Loading DenseNet121 ImageNet weights...")
        model = build_densenet121_feature_extractor(device)
    else:
        print(f"Loading DINOv3 model: {args.model_name}")
        model = build_dinov3_feature_extractor(
            model_name=args.model_name,
            device=device,
            hf_token=args.hf_token,
        )

    for split_csv in args.splits:
        split_csv = Path(split_csv)
        if not split_csv.exists():
            raise FileNotFoundError(f"CSV not found: {split_csv}")

        print(f"\nExtracting features from: {split_csv}")
        dataloader = build_dataloader(
            csv_path=split_csv,
            data_root=args.data_root,
            image_size=args.image_size,
            batch_size=batch_size,
            num_workers=args.num_workers,
            shuffle=False,
        )

        feature_data = extract_features(
            model=model,
            dataloader=dataloader,
            device=device,
            model_type=args.model,
            feature_type=args.feature_type,
            max_batches=args.max_batches,
        )

        output_path = output_path_for_split(
            csv_path=split_csv,
            output_dir=output_dir,
            model_type=args.model,
            model_name=args.model_name,
            feature_type=args.feature_type,
        )
        torch.save(feature_data, output_path)

        print(f"Saved: {output_path}")
        print(f"  features: {tuple(feature_data['features'].shape)}")
        print(f"  labels: {tuple(feature_data['labels'].shape)}")
        print(f"  paths: {len(feature_data['paths'])}")


if __name__ == "__main__":
    main()
