"""Backbones, classifier heads, and feature pooling.

Follows the jfhealthcare/Chexpert convention of a small set of classifier
wrappers over swappable backbones, plus the DINO feature-pooling helpers shared
between frozen feature extraction and end-to-end fine-tuning.
"""

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import DenseNet121_Weights, densenet121

from chexpert.data import TARGET_LABELS


# --------------------------------------------------------------------------- #
# Feature pooling (shared by extraction and fine-tuning)
# --------------------------------------------------------------------------- #
def select_densenet121_features(model, images):
    features = model.features(images)
    features = F.relu(features, inplace=False)
    features = F.adaptive_avg_pool2d(features, (1, 1))
    return torch.flatten(features, 1)


def pool_dino_features(outputs, config, feature_type):
    if feature_type == "pooled":
        if getattr(outputs, "pooler_output", None) is None:
            raise ValueError("This model output does not contain pooler_output.")
        return outputs.pooler_output

    hidden_states = outputs.last_hidden_state

    if feature_type == "cls":
        return hidden_states[:, 0, :]

    num_register_tokens = getattr(config, "num_register_tokens", 0)
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


# --------------------------------------------------------------------------- #
# Classifier heads
# --------------------------------------------------------------------------- #
class LinearHead(nn.Module):
    def __init__(self, input_dim, num_labels):
        super().__init__()
        self.classifier = nn.Linear(input_dim, num_labels)

    def forward(self, features):
        return self.classifier(features)


class MLPHead(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_labels, dropout):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_labels),
        )

    def forward(self, features):
        return self.classifier(features)


def build_head(head_type, input_dim, hidden_dim, num_labels, dropout):
    if head_type == "linear":
        return LinearHead(input_dim=input_dim, num_labels=num_labels)
    if head_type == "mlp":
        return MLPHead(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_labels=num_labels,
            dropout=dropout,
        )
    raise ValueError(f"Unknown head_type: {head_type}")


def build_head_model(checkpoint):
    """Reconstruct a frozen-feature head from a saved checkpoint dict."""
    input_dim = checkpoint["input_dim"]
    num_labels = checkpoint["num_labels"]

    if "hidden_dim" in checkpoint:
        return MLPHead(
            input_dim=input_dim,
            hidden_dim=checkpoint["hidden_dim"],
            num_labels=num_labels,
            dropout=checkpoint.get("dropout", 0.0),
        )

    return LinearHead(input_dim=input_dim, num_labels=num_labels)


# --------------------------------------------------------------------------- #
# End-to-end classifiers (backbone + head)
# --------------------------------------------------------------------------- #
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
        features = select_densenet121_features(self.backbone, images)
        return self.classifier(features)


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
        return pool_dino_features(outputs, self.backbone.config, self.feature_type)

    def forward(self, pixel_values):
        outputs = self.backbone(pixel_values=pixel_values)
        features = self.select_features(outputs)
        return self.classifier(features)


# --------------------------------------------------------------------------- #
# Backbone loading
# --------------------------------------------------------------------------- #
def build_densenet121_feature_extractor(device):
    weights = DenseNet121_Weights.DEFAULT
    model = densenet121(weights=weights)
    model.classifier = torch.nn.Identity()
    model.eval()
    model.to(device)
    return model


def build_dinov3_feature_extractor(model_name, device, hf_token=None):
    try:
        from transformers import AutoModel
    except ImportError as exc:
        raise ImportError(
            "transformers is required for DINOv3 feature extraction. "
            "Install it with: pip install transformers"
        ) from exc

    model = AutoModel.from_pretrained(
        model_name,
        token=hf_token,
    )
    model.eval()
    model.to(device)
    return model


def load_dinov3_backbone(model_name, hf_token):
    """Load a raw DINOv3 backbone (no freezing/eval), for fine-tuning/evaluation."""
    try:
        from transformers import AutoModel
    except ImportError as exc:
        raise ImportError("Install transformers first: pip install transformers") from exc

    return AutoModel.from_pretrained(model_name, token=hf_token)


# --------------------------------------------------------------------------- #
# Parameter freezing / unfreezing
# --------------------------------------------------------------------------- #
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
    return "features.denseblock4 + features.norm5", trainable_backbone, total_backbone


# --------------------------------------------------------------------------- #
# Fine-tuned checkpoint reconstruction
# --------------------------------------------------------------------------- #
def build_finetuned_model(checkpoint, device, hf_token):
    model_name = checkpoint.get("model_name", "")
    args = checkpoint.get("args", {})
    head_type = checkpoint.get("head_type", args.get("head_type", "mlp"))
    hidden_dim = checkpoint.get("hidden_dim", args.get("hidden_dim", 512))
    dropout = checkpoint.get("dropout", args.get("dropout", 0.0))

    if "dinov3" in model_name:
        backbone = load_dinov3_backbone(model_name, hf_token)
        model = DINOv3Classifier(
            backbone=backbone,
            feature_type=checkpoint.get("feature_type", args.get("feature_type")),
            head_type=head_type,
            hidden_dim=hidden_dim,
            dropout=dropout,
            num_labels=len(TARGET_LABELS),
        )
    elif model_name == "densenet121_imagenet":
        model = DenseNet121Classifier(
            head_type=head_type,
            hidden_dim=hidden_dim,
            dropout=dropout,
            num_labels=len(TARGET_LABELS),
        )
    else:
        raise ValueError(f"Unsupported fine-tuned checkpoint model_name: {model_name}")

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


# Backwards-compatible alias used by the CheXlocalize panel exporter.
build_model_from_checkpoint = build_finetuned_model
