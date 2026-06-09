import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch

from chexpert_data import TARGET_LABELS


def torch_load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_feature_file(path):
    data = torch_load(path)
    features = data["features"].float()
    labels = data["labels"].float()
    paths = data.get("paths", [""] * features.shape[0])
    model_name = data.get("feature_model", Path(path).stem)
    feature_type = data.get("feature_type")

    return {
        "features": features,
        "labels": labels,
        "paths": paths,
        "model_name": model_name,
        "feature_type": feature_type,
    }


def sample_indices(labels, max_samples, seed):
    n = labels.shape[0]
    if n <= max_samples:
        return torch.arange(n)

    generator = torch.Generator().manual_seed(seed)
    return torch.randperm(n, generator=generator)[:max_samples]


def standardize_features(features, eps=1e-6):
    mean = features.mean(dim=0, keepdim=True)
    std = features.std(dim=0, keepdim=True, unbiased=False).clamp_min(eps)
    return (features - mean) / std


def label_values(labels, color_by):
    if color_by == "primary":
        positive_counts = labels.sum(dim=1)
        values = labels.argmax(dim=1)
        values[positive_counts == 0] = -1
        names = ["No positive"] + TARGET_LABELS
        return values, names

    if color_by == "any_positive":
        values = (labels.sum(dim=1) > 0).long()
        return values, ["No positive", "Any positive"]

    if color_by not in TARGET_LABELS:
        raise ValueError(
            f"--color-by must be one of primary, any_positive, or {TARGET_LABELS}"
        )

    label_idx = TARGET_LABELS.index(color_by)
    values = labels[:, label_idx].long()
    return values, [f"{color_by}=0", f"{color_by}=1"]


def fit_umap(features, n_neighbors, min_dist, metric, seed):
    try:
        import umap
    except ImportError as exc:
        raise ImportError(
            "umap-learn is required. Install it with: pip install umap-learn"
        ) from exc

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=seed,
    )
    return reducer.fit_transform(features.numpy())


def plot_embedding(ax, embedding, values, class_names, title):
    unique_values = sorted(set(values.tolist()))
    cmap = plt.get_cmap("tab10")

    for plot_idx, value in enumerate(unique_values):
        mask = values.numpy() == value
        if value == -1:
            label = "No positive"
            color = "lightgray"
        elif len(class_names) == 2 and class_names[0].endswith("=0"):
            label = class_names[value]
            color = cmap(plot_idx % 10)
        elif class_names == ["No positive", "Any positive"]:
            label = class_names[value]
            color = "lightgray" if value == 0 else cmap(1)
        else:
            label = class_names[value + 1] if "No positive" in class_names else class_names[value]
            color = cmap(plot_idx % 10)

        ax.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            s=7,
            alpha=0.65,
            linewidths=0,
            label=label,
            color=color,
        )

    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(markerscale=2, fontsize=8, loc="best")


def write_coordinates(path, embedding, values, labels, image_paths):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["umap_x", "umap_y", "color_value", "path"] + TARGET_LABELS
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for idx in range(embedding.shape[0]):
            row = {
                "umap_x": float(embedding[idx, 0]),
                "umap_y": float(embedding[idx, 1]),
                "color_value": int(values[idx].item()),
                "path": image_paths[idx],
            }
            for label_idx, label_name in enumerate(TARGET_LABELS):
                row[label_name] = int(labels[idx, label_idx].item())
            writer.writerow(row)


def safe_name(text):
    return (
        str(text)
        .replace("/", "_")
        .replace("\\", "_")
        .replace(" ", "_")
        .replace(":", "_")
    )


def normalize_path_key(path_value):
    path_value = str(path_value).replace("\\", "/")
    prefix = "CheXpert-v1.0-small/"
    data_prefix = "data/"

    if path_value.startswith(prefix):
        return path_value[len(prefix) :]
    if path_value.startswith(data_prefix):
        return path_value[len(data_prefix) :]

    parts = path_value.split("/")
    for split_name in ("train", "valid"):
        if split_name in parts:
            return "/".join(parts[parts.index(split_name) :])

    return path_value


def load_metadata(csv_paths):
    if not csv_paths:
        return {}

    frames = []
    for csv_path in csv_paths:
        frame = pd.read_csv(csv_path)
        if "Path" not in frame.columns:
            raise ValueError(f"CSV must contain Path column: {csv_path}")
        frames.append(frame)

    df = pd.concat(frames, ignore_index=True)
    df["_path_key"] = df["Path"].apply(normalize_path_key)
    return df.drop_duplicates("_path_key").set_index("_path_key").to_dict("index")


def filter_by_metadata(features, labels, paths, metadata, filters):
    if not filters:
        return features, labels, paths

    keep_indices = []
    missing = 0
    for idx, path in enumerate(paths):
        row = metadata.get(normalize_path_key(path))
        if row is None:
            missing += 1
            continue
        if all(str(row.get(column)) == str(value) for column, value in filters):
            keep_indices.append(idx)

    if not keep_indices:
        filter_text = ", ".join(f"{column}={value}" for column, value in filters)
        raise ValueError(
            f"No samples matched metadata filter {filter_text}. "
            f"Missing metadata rows: {missing}"
        )

    idx_tensor = torch.tensor(keep_indices, dtype=torch.long)
    return (
        features[idx_tensor],
        labels[idx_tensor],
        [paths[idx] for idx in keep_indices],
    )


def main():
    parser = argparse.ArgumentParser(
        description="Plot UMAP embeddings from saved DenseNet/DINO feature files."
    )
    parser.add_argument(
        "--feature-files",
        nargs="+",
        required=True,
        help="One or more .pt feature files.",
    )
    parser.add_argument("--output-dir", default="umap_outputs")
    parser.add_argument(
        "--color-by",
        default="primary",
        help="primary, any_positive, or one target label name.",
    )
    parser.add_argument("--max-samples", type=int, default=5000)
    parser.add_argument("--n-neighbors", type=int, default=15)
    parser.add_argument("--min-dist", type=float, default=0.1)
    parser.add_argument("--metric", default="cosine")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-standardize", action="store_true")
    parser.add_argument(
        "--metadata-csvs",
        nargs="*",
        default=None,
        help="CSV files containing Path and metadata columns such as Frontal/Lateral.",
    )
    parser.add_argument(
        "--filter-column",
        default=None,
        help="Metadata column used to filter samples before UMAP.",
    )
    parser.add_argument(
        "--filter-value",
        default=None,
        help="Metadata value to keep, for example Frontal or Lateral.",
    )
    parser.add_argument(
        "--filter",
        action="append",
        default=None,
        help=(
            "Metadata filter in COLUMN=VALUE format. Can be used multiple times, "
            "for example --filter Frontal/Lateral=Frontal --filter AP/PA=AP."
        ),
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = load_metadata(args.metadata_csvs)

    filters = []
    if args.filter_column:
        if args.filter_value is None:
            raise ValueError("--filter-column requires --filter-value.")
        filters.append((args.filter_column, args.filter_value))

    if args.filter:
        for filter_item in args.filter:
            if "=" not in filter_item:
                raise ValueError("--filter entries must use COLUMN=VALUE format.")
            column, value = filter_item.split("=", 1)
            filters.append((column, value))

    if filters and not metadata:
        raise ValueError("Metadata filters require --metadata-csvs.")

    loaded = []
    for feature_file in args.feature_files:
        feature_file = Path(feature_file)
        data = load_feature_file(feature_file)
        features, labels, paths = filter_by_metadata(
            features=data["features"],
            labels=data["labels"],
            paths=data["paths"],
            metadata=metadata,
            filters=filters,
        )
        indices = sample_indices(labels, args.max_samples, args.seed)

        features = features[indices]
        labels = labels[indices]
        image_paths = [paths[idx] for idx in indices.tolist()]

        if not args.no_standardize:
            features = standardize_features(features)

        values, class_names = label_values(labels, args.color_by)
        print(f"Fitting UMAP: {feature_file} samples={features.shape[0]}")
        embedding = fit_umap(
            features=features,
            n_neighbors=args.n_neighbors,
            min_dist=args.min_dist,
            metric=args.metric,
            seed=args.seed,
        )

        title = str(data["model_name"])
        if data["feature_type"]:
            title = f"{title} ({data['feature_type']})"

        loaded.append(
            {
                "feature_file": feature_file,
                "title": title,
                "embedding": embedding,
                "values": values,
                "labels": labels,
                "paths": image_paths,
                "class_names": class_names,
            }
        )

        stem = safe_name(feature_file.stem)
        write_coordinates(
            output_dir / f"{stem}_umap_coordinates.csv",
            embedding,
            values,
            labels,
            image_paths,
        )

        fig, ax = plt.subplots(figsize=(7, 6))
        plot_embedding(ax, embedding, values, class_names, title)
        fig.tight_layout()
        fig.savefig(output_dir / f"{stem}_umap.png", dpi=220)
        plt.close(fig)

    if len(loaded) > 1:
        fig, axes = plt.subplots(1, len(loaded), figsize=(7 * len(loaded), 6))
        if len(loaded) == 1:
            axes = [axes]

        for ax, item in zip(axes, loaded):
            plot_embedding(
                ax,
                item["embedding"],
                item["values"],
                item["class_names"],
                item["title"],
            )

        fig.suptitle(f"UMAP Comparison - color by {args.color_by}")
        fig.tight_layout()
        fig.savefig(output_dir / f"umap_comparison_color_by_{safe_name(args.color_by)}.png", dpi=220)
        plt.close(fig)

    print(f"Saved UMAP outputs to: {output_dir}")


if __name__ == "__main__":
    main()
