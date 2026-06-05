"""Prepare CheXpert CSVs: patient-wise split + U-Zeros relabeling."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chexpert.prepare import preprocess_splits, split_patientwise


def main():
    parser = argparse.ArgumentParser(description="Prepare CheXpert CSV files.")
    subparsers = parser.add_subparsers(dest="command")

    split_parser = subparsers.add_parser(
        "split",
        help="Split CheXpert train.csv into train/intraval CSVs by patient id.",
    )
    split_parser.add_argument("--input", default="data/train.csv")
    split_parser.add_argument("--train-output", default="data/train_internal.csv")
    split_parser.add_argument("--intraval-output", default="data/intraval_internal.csv")
    split_parser.add_argument("--train-ratio", type=float, default=0.9)
    split_parser.add_argument("--seed", type=int, default=42)

    uzeros_parser = subparsers.add_parser(
        "uzeros",
        help="Apply U-Zeros label preprocessing for the 5 CheXpert labels.",
    )
    uzeros_parser.add_argument(
        "--splits",
        nargs="+",
        default=[
            "data/train_internal.csv",
            "data/intraval_internal.csv",
            "data/valid.csv",
        ],
    )
    uzeros_parser.add_argument("--output-dir", default="data/processed")

    all_parser = subparsers.add_parser(
        "all",
        help="Run patient-wise split followed by U-Zeros preprocessing.",
    )
    all_parser.add_argument("--input", default="data/train.csv")
    all_parser.add_argument("--train-output", default="data/train_internal.csv")
    all_parser.add_argument("--intraval-output", default="data/intraval_internal.csv")
    all_parser.add_argument("--valid-input", default="data/valid.csv")
    all_parser.add_argument("--processed-output-dir", default="data/processed")
    all_parser.add_argument("--train-ratio", type=float, default=0.9)
    all_parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    if args.command is None:
        split_patientwise(
            input_path="data/train.csv",
            train_output="data/train_internal.csv",
            intraval_output="data/intraval_internal.csv",
            train_ratio=0.9,
            seed=42,
        )
        return

    if args.command == "split":
        split_patientwise(
            input_path=args.input,
            train_output=args.train_output,
            intraval_output=args.intraval_output,
            train_ratio=args.train_ratio,
            seed=args.seed,
        )
    elif args.command == "uzeros":
        preprocess_splits(args.splits, args.output_dir)
    elif args.command == "all":
        split_patientwise(
            input_path=args.input,
            train_output=args.train_output,
            intraval_output=args.intraval_output,
            train_ratio=args.train_ratio,
            seed=args.seed,
        )
        preprocess_splits(
            [args.train_output, args.intraval_output, args.valid_input],
            args.processed_output_dir,
        )


if __name__ == "__main__":
    main()
