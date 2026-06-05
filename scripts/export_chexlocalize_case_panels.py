"""Export 4-panel CheXlocalize cases: original, GT annotation, and the
DenseNet / DINO input-gradient saliency maps side by side."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from PIL import Image
from matplotlib.patches import Polygon
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from chexpert.data import IMAGENET_MEAN, IMAGENET_STD, TARGET_LABELS
from chexpert.engine import torch_load
from chexpert.models import build_model_from_checkpoint


DEFAULT_CASES = [
    ("Consolidation", "patient64624_study1_view1_frontal", "DINO higher confidence"),
    ("Atelectasis", "patient64548_study1_view1_frontal", "DenseNet higher confidence"),
    ("Consolidation", "patient64688_study1_view1_frontal", "High-confidence TP"),
]


def key_from_path(path):
    path = Path(path)
    return f"{path.parts[-3]}_{path.parts[-2]}_{path.stem}"


def image_path_from_key(key, image_root):
    patient, study, *view_parts = key.split("_")
    view = "_".join(view_parts)
    return image_root / patient / study / f"{view}.jpg"


def load_prediction_map(prediction_path):
    data = torch_load(prediction_path)
    key_to_index = {key_from_path(path): idx for idx, path in enumerate(data["paths"])}
    return data, key_to_index


def preprocess_image(image, image_size, device):
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    tensor = transform(image).unsqueeze(0).to(device)
    return tensor


def normalize_heatmap(heatmap):
    heatmap = heatmap.detach().float().cpu()
    high = torch.quantile(heatmap.flatten(), 0.99)
    heatmap = heatmap.clamp(max=high)
    heatmap = heatmap - heatmap.min()
    max_value = heatmap.max().clamp_min(1e-8)
    return heatmap / max_value


def input_gradient_saliency(model, image_tensor, label_idx):
    model.eval()
    x = image_tensor.detach().clone().requires_grad_(True)
    model.zero_grad(set_to_none=True)
    logits = model(x)
    score = logits[0, label_idx]
    score.backward()
    heatmap = x.grad.detach().abs().amax(dim=1)[0]
    return normalize_heatmap(heatmap), float(torch.sigmoid(score).detach().cpu())


def draw_gt_overlay(ax, image, polygons, disease):
    ax.imshow(image)
    ax.set_title(f"GT: {disease}")
    ax.axis("off")
    for polygon in polygons:
        patch = Polygon(
            polygon,
            closed=True,
            facecolor=(1.0, 0.15, 0.05, 0.25),
            edgecolor=(1.0, 0.05, 0.0, 0.95),
            linewidth=2.0,
        )
        ax.add_patch(patch)


def draw_heatmap(ax, image, heatmap, title):
    ax.imshow(image)
    ax.imshow(heatmap, cmap="turbo", alpha=0.58, extent=(0, image.width, image.height, 0))
    ax.set_title(title)
    ax.axis("off")


def draw_case_panel(
    image_path,
    polygons,
    disease,
    title,
    densenet_heatmap,
    dino_heatmap,
    densenet_prob,
    dino_prob,
    output_path,
):
    image = Image.open(image_path).convert("RGB")
    densenet_heatmap = F.interpolate(
        densenet_heatmap[None, None, :, :],
        size=(image.height, image.width),
        mode="bilinear",
        align_corners=False,
    )[0, 0]
    dino_heatmap = F.interpolate(
        dino_heatmap[None, None, :, :],
        size=(image.height, image.width),
        mode="bilinear",
        align_corners=False,
    )[0, 0]

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    axes[0].imshow(image)
    axes[0].set_title("Original")
    axes[0].axis("off")

    draw_gt_overlay(axes[1], image, polygons, disease)
    draw_heatmap(
        axes[2],
        image,
        densenet_heatmap,
        f"DenseNet input-gradient saliency\np={densenet_prob:.3f}",
    )
    draw_heatmap(
        axes[3],
        image,
        dino_heatmap,
        f"DINO input-gradient saliency\np={dino_prob:.3f}",
    )

    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Export 4-panel CheXlocalize cases: original, GT annotation, "
            "DenseNet input-gradient saliency, and DINO input-gradient saliency."
        )
    )
    parser.add_argument(
        "--annotation-json",
        default="data_download/chexlocalize/CheXlocalize/gt_annotations_val.json",
    )
    parser.add_argument("--image-root", default="data_download/chexlocalize/CheXpert/val")
    parser.add_argument(
        "--dino-checkpoint",
        default="checkpoints/dinov3_last_block_mean_patch_mlp_30ep_bs16_w4_h512_drop01_best.pt",
    )
    parser.add_argument(
        "--densenet-checkpoint",
        default="checkpoints/densenet121_denseblock4_mlp_30ep_bs16_w4_h512_drop01_best.pt",
    )
    parser.add_argument(
        "--dino-predictions",
        default=(
            "evaluation/finetuned/"
            "dinov3_last_block_mean_patch_mlp_30ep_bs16_w4_h512_drop01_valid/"
            "predictions.pt"
        ),
    )
    parser.add_argument(
        "--densenet-predictions",
        default=(
            "evaluation/finetuned/"
            "densenet121_denseblock4_mlp_30ep_bs16_w4_h512_drop01_valid/"
            "predictions.pt"
        ),
    )
    parser.add_argument("--output-dir", default="chexlocalize_case_panels")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--hf-token", default=None)
    args = parser.parse_args()

    device = torch.device(args.device)
    annotations = json.load(open(args.annotation_json, encoding="utf-8"))
    image_root = Path(args.image_root)
    output_dir = Path(args.output_dir)

    dino_predictions, dino_index = load_prediction_map(args.dino_predictions)
    densenet_predictions, densenet_index = load_prediction_map(args.densenet_predictions)

    dino_checkpoint = torch_load(args.dino_checkpoint, map_location=device)
    densenet_checkpoint = torch_load(args.densenet_checkpoint, map_location=device)
    dino_model = build_model_from_checkpoint(dino_checkpoint, device, args.hf_token)
    densenet_model = build_model_from_checkpoint(densenet_checkpoint, device, args.hf_token)

    for disease, key, note in DEFAULT_CASES:
        if key not in annotations:
            raise KeyError(f"Missing annotation key: {key}")
        if disease not in annotations[key] or not annotations[key][disease]:
            raise KeyError(f"Missing {disease} annotation for {key}")
        if key not in dino_index or key not in densenet_index:
            raise KeyError(f"Missing predictions for {key}")

        label_idx = TARGET_LABELS.index(disease)
        image_path = image_path_from_key(key, image_root)
        image = Image.open(image_path).convert("RGB")
        image_tensor = preprocess_image(image, args.image_size, device)

        densenet_heatmap, densenet_saliency_prob = input_gradient_saliency(
            densenet_model, image_tensor, label_idx
        )
        dino_heatmap, dino_saliency_prob = input_gradient_saliency(
            dino_model, image_tensor, label_idx
        )

        dino_prob = float(dino_predictions["probs"][dino_index[key], label_idx])
        densenet_prob = float(densenet_predictions["probs"][densenet_index[key], label_idx])
        true_label = int(dino_predictions["labels"][dino_index[key], label_idx].item())

        title = (
            f"{key} | {disease} | y={true_label} | "
            f"DINO={dino_prob:.3f}, DenseNet={densenet_prob:.3f} | {note}"
        )
        if abs(dino_prob - dino_saliency_prob) > 1e-3:
            print(f"Warning: DINO probability mismatch for {key}")
        if abs(densenet_prob - densenet_saliency_prob) > 1e-3:
            print(f"Warning: DenseNet probability mismatch for {key}")

        output_name = f"{key}_{disease.replace(' ', '_')}_4panel.png"
        draw_case_panel(
            image_path=image_path,
            polygons=annotations[key][disease],
            disease=disease,
            title=title,
            densenet_heatmap=densenet_heatmap,
            dino_heatmap=dino_heatmap,
            densenet_prob=densenet_prob,
            dino_prob=dino_prob,
            output_path=output_dir / output_name,
        )
        print(output_dir / output_name)


if __name__ == "__main__":
    main()
