"""CheXpert dataset, transforms, and the canonical 5-label list.

Single source of truth for ``TARGET_LABELS`` and the ImageNet normalization
constants, mirroring how ``torchxrayvision.datasets`` keeps the pathology list
alongside the dataset definition.
"""

from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode


TARGET_LABELS = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Pleural Effusion",
]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_image_transform(image_size=224):
    return transforms.Compose(
        [
            transforms.Resize(
                (image_size, image_size),
                interpolation=InterpolationMode.BICUBIC,
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def make_u_zeros_labels(df):
    missing_labels = [label for label in TARGET_LABELS if label not in df.columns]
    if missing_labels:
        raise ValueError(f"Missing target label columns: {missing_labels}")

    labels = df[TARGET_LABELS].copy()
    labels = labels.fillna(0).replace(-1, 0)
    return labels.astype("float32")


def resolve_chexpert_path(path_value, data_root):
    path_value = str(path_value)
    prefix = "CheXpert-v1.0-small/"

    if path_value.startswith(prefix):
        relative_path = path_value[len(prefix) :]
    else:
        relative_path = path_value

    return Path(data_root) / relative_path


class CheXpertFiveLabelDataset(Dataset):
    def __init__(self, csv_path, data_root="data", transform=None):
        self.csv_path = Path(csv_path)
        self.data_root = Path(data_root)
        self.transform = transform or get_image_transform()

        self.df = pd.read_csv(self.csv_path)
        if "Path" not in self.df.columns:
            raise ValueError("CSV must contain a 'Path' column.")

        self.labels = make_u_zeros_labels(self.df).to_numpy(dtype="float32")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        image_path = resolve_chexpert_path(row["Path"], self.data_root)

        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        label = torch.tensor(self.labels[index], dtype=torch.float32)
        return image, label, str(image_path)


def build_dataloader(
    csv_path,
    data_root="data",
    image_size=224,
    batch_size=32,
    num_workers=0,
    shuffle=False,
):
    dataset = CheXpertFiveLabelDataset(
        csv_path=csv_path,
        data_root=data_root,
        transform=get_image_transform(image_size=image_size),
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def check_dataset(csv_path, data_root, image_size, max_missing):
    dataset = CheXpertFiveLabelDataset(
        csv_path=csv_path,
        data_root=data_root,
        transform=get_image_transform(image_size=image_size),
    )

    missing_paths = []
    for path_value in dataset.df["Path"]:
        image_path = resolve_chexpert_path(path_value, data_root)
        if not image_path.exists():
            missing_paths.append(str(image_path))
            if len(missing_paths) >= max_missing:
                break

    if missing_paths:
        print(f"Missing images found: {len(missing_paths)} shown")
        for path in missing_paths:
            print(path)
        raise FileNotFoundError("At least one image path from the CSV does not exist.")

    image, label, image_path = dataset[0]
    print(f"CSV: {csv_path}")
    print(f"Rows: {len(dataset)}")
    print(f"First image path: {image_path}")
    print(f"Image tensor shape: {tuple(image.shape)}")
    print(f"Image tensor dtype: {image.dtype}")
    print(f"Label shape: {tuple(label.shape)}")
    print(f"First U-Zeros label: {label.tolist()}")
    print("Preprocessing check passed.")
