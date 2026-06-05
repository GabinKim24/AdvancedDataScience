"""Sanity-check a CheXpert CSV: verify image paths resolve and show one sample."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chexpert.data import check_dataset


def main():
    parser = argparse.ArgumentParser(
        description="CheXpert preprocessing utilities for 5-label classification."
    )
    parser.add_argument("--csv", default="data/train_internal.csv")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--max-missing", type=int, default=10)
    args = parser.parse_args()

    check_dataset(
        csv_path=args.csv,
        data_root=args.data_root,
        image_size=args.image_size,
        max_missing=args.max_missing,
    )


if __name__ == "__main__":
    main()
