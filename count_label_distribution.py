import argparse
from pathlib import Path

import pandas as pd

from chexpert_data import TARGET_LABELS


def count_label_values(csv_path, split_name):
    df = pd.read_csv(csv_path)

    results = []
    for label in TARGET_LABELS:
        if label not in df.columns:
            raise ValueError(f"Column '{label}' not found in {csv_path}")

        col = df[label]
        results.append(
            {
                "split": split_name,
                "label": label,
                "positive(1)": int((col == 1).sum()),
                "negative(0)": int((col == 0).sum()),
                "uncertain(-1)": int((col == -1).sum()),
                "blank(NaN)": int(col.isna().sum()),
                "total": int(len(col)),
            }
        )

    return pd.DataFrame(results)


def count_combined_label_values(csv_paths):
    dfs = [pd.read_csv(path) for path in csv_paths]
    df = pd.concat(dfs, ignore_index=True)

    results = []
    for label in TARGET_LABELS:
        col = df[label]
        results.append(
            {
                "label": label,
                "positive(1)": int((col == 1).sum()),
                "negative(0)": int((col == 0).sum()),
                "uncertain(-1)": int((col == -1).sum()),
                "blank(NaN)": int(col.isna().sum()),
                "total": int(len(col)),
            }
        )

    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(
        description="Count CheXpert 5-label value distribution per split and combined."
    )
    parser.add_argument("--train-csv", default="data/train.csv")
    parser.add_argument("--valid-csv", default="data/valid.csv")
    parser.add_argument(
        "--output",
        default="results/chexpert_5label_distribution_combined.csv",
    )
    args = parser.parse_args()

    train_counts = count_label_values(args.train_csv, "train")
    valid_counts = count_label_values(args.valid_csv, "valid")
    per_split = pd.concat([train_counts, valid_counts], ignore_index=True)
    print(per_split.to_string(index=False))

    combined = count_combined_label_values([args.train_csv, args.valid_csv])
    print()
    print(combined.to_string(index=False))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path, index=False)
    print(f"\nSaved combined distribution to: {output_path}")


if __name__ == "__main__":
    main()
