"""Apply the U-Zeros label policy to CheXpert split CSVs.

Provenance (see PROVENANCE.md): the U-Zeros policy (blanks -> 0, uncertain -1
-> 0 for the 5 competition labels) is adapted from
MohammedSB/DINOv2ForRadiology `dinov2/data/datasets/chexpert.py` (CC BY-NC 4.0)
and matches Stomper10/CheXpert's "zeroes" uncertainty option (approach informed
by; no code copied). The CSV I/O wrapper is the author's own (USED-OWN).
"""

import argparse
from pathlib import Path

import pandas as pd

from chexpert_data import TARGET_LABELS


def apply_u_zeros(df):
    missing_labels = [label for label in TARGET_LABELS if label not in df.columns]
    if missing_labels:
        raise ValueError(f"Missing target label columns: {missing_labels}")

    processed_df = df.copy()
    processed_df[TARGET_LABELS] = (
        processed_df[TARGET_LABELS]
        .fillna(0)
        .replace(-1, 0)
        .astype("float32")
    )

    return processed_df


def summarize_labels(df, split_name):
    print(f"\n[{split_name}]")
    print(f"rows: {len(df)}")

    for label in TARGET_LABELS:
        positives = int((df[label] == 1).sum())
        negatives = int((df[label] == 0).sum())
        print(f"{label}: positive={positives}, negative={negatives}")


def preprocess_csv(input_path, output_path):
    input_path = Path(input_path)
    output_path = Path(output_path)

    df = pd.read_csv(input_path)
    processed_df = apply_u_zeros(df)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    processed_df.to_csv(output_path, index=False)

    print(f"Saved: {output_path}")
    summarize_labels(processed_df, output_path.stem)


def main():
    parser = argparse.ArgumentParser(
        description="Apply U-Zeros label preprocessing for CheXpert 5-label task."
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=[
            "data/train_internal.csv",
            "data/intraval_internal.csv",
            "data/valid.csv",
        ],
        help="Input CSV files to preprocess.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed",
        help="Directory where U-Zeros CSV files will be saved.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    for split_path in args.splits:
        split_path = Path(split_path)
        output_path = output_dir / f"{split_path.stem}_uzeros.csv"
        preprocess_csv(split_path, output_path)


if __name__ == "__main__":
    main()
