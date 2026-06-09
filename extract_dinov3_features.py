import argparse
import os
from pathlib import Path

import torch

from chexpert_data import TARGET_LABELS, build_dataloader


DEFAULT_SPLITS = [
    "data/processed/train_internal_uzeros.csv",
    "data/processed/intraval_internal_uzeros.csv",
    "data/processed/valid_uzeros.csv",
]


def load_dinov3_model(model_name, device, hf_token=None):
    try:
        from transformers import AutoModel
    except ImportError as exc:
        raise ImportError(
            "transformers is required for DINOv3 feature extraction. "
            "Install it with: pip install transformers"
        ) from exc

    model = AutoModel.from_pretrained(model_name, token=hf_token)
    model.eval()
    model.to(device)
    return model


def select_features(outputs, model, feature_type):
    if feature_type == "pooled":
        if getattr(outputs, "pooler_output", None) is None:
            raise ValueError("This model output does not contain pooler_output.")
        return outputs.pooler_output

    hidden_states = outputs.last_hidden_state

    if feature_type == "cls":
        return hidden_states[:, 0, :]

    num_register_tokens = getattr(model.config, "num_register_tokens", 0)
    patch_start = 1 + num_register_tokens
    patch_tokens = hidden_states[:, patch_start:, :]

    if feature_type == "mean_patch":
        return patch_tokens.mean(dim=1)
    if feature_type == "max_patch":
        return patch_tokens.max(dim=1).values
    if feature_type == "mean_max_patch":
        mean_patch = patch_tokens.mean(dim=1)
        max_patch = patch_tokens.max(dim=1).values
        return torch.cat([mean_patch, max_patch], dim=1)

    raise ValueError(f"Unknown feature_type: {feature_type}")


def extract_features(model, dataloader, device, feature_type, max_batches=None):
    all_features = []
    all_labels = []
    all_paths = []

    with torch.inference_mode():
        for batch_idx, (images, labels, paths) in enumerate(dataloader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            images = images.to(device, non_blocking=True)
            outputs = model(pixel_values=images)
            features = select_features(outputs, model, feature_type)

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
        "feature_model": model.config.name_or_path,
        "feature_type": feature_type,
    }


def safe_model_name(model_name):
    return model_name.replace("/", "_").replace("-", "_")


def output_path_for_split(csv_path, output_dir, model_name, feature_type):
    csv_path = Path(csv_path)
    model_tag = safe_model_name(model_name)
    return Path(output_dir) / f"{csv_path.stem}_{model_tag}_{feature_type}_features.pt"


def main():
    parser = argparse.ArgumentParser(
        description="Extract DINOv3 features for CheXpert CSV splits."
    )
    parser.add_argument("--splits", nargs="+", default=DEFAULT_SPLITS)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--output-dir", default="features")
    parser.add_argument(
        "--model-name",
        default="facebook/dinov3-vitb16-pretrain-lvd1689m",
        help="Hugging Face DINOv3 model id.",
    )
    parser.add_argument(
        "--feature-type",
        default="mean_patch",
        choices=["pooled", "cls", "mean_patch", "max_patch", "mean_max_patch"],
        help="Global feature representation to save.",
    )
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
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
    print(f"Loading DINOv3 model: {args.model_name}")
    model = load_dinov3_model(
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
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=False,
        )

        feature_data = extract_features(
            model=model,
            dataloader=dataloader,
            device=device,
            feature_type=args.feature_type,
            max_batches=args.max_batches,
        )

        output_path = output_path_for_split(
            csv_path=split_csv,
            output_dir=output_dir,
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
