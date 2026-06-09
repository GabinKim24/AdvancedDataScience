"""Patient-wise split of CheXpert train.csv into intra-train / intra-val CSVs.

Provenance (see PROVENANCE.md): author's own original code (USED-OWN). The
patient-wise (group-by-patient) split avoids study/patient leakage between the
internal train and validation folds; this is the project's own design and is
NOT taken from Stomper10 (which splits by frontal/lateral and uses the official
valid set as test) nor from DINOv2ForRadiology.
"""

import argparse
import re
from pathlib import Path

import pandas as pd

from chexpert_data import TARGET_LABELS


def extract_patient_id(path_value):
    match = re.search(r"(patient\d+)", str(path_value))
    if match is None:
        raise ValueError(f"Could not extract patient id from Path: {path_value}")
    return match.group(1)


def u_zeros_labels(df):
    labels = df[TARGET_LABELS].copy()
    labels = labels.fillna(0).replace(-1, 0)
    return labels.astype("float32")


def patientwise_split(df, train_ratio, seed):
    df = df.copy()
    df["_patient_id"] = df["Path"].apply(extract_patient_id)

    patients = pd.Series(df["_patient_id"].unique()).sample(
        frac=1.0,
        random_state=seed,
    )

    n_train_patients = round(len(patients) * train_ratio)
    train_patients = set(patients.iloc[:n_train_patients])
    intraval_patients = set(patients.iloc[n_train_patients:])

    train_df = df[df["_patient_id"].isin(train_patients)].copy()
    intraval_df = df[df["_patient_id"].isin(intraval_patients)].copy()

    overlap = set(train_df["_patient_id"]) & set(intraval_df["_patient_id"])
    if overlap:
        raise RuntimeError(f"Patient leakage detected: {sorted(overlap)[:5]}")

    return (
        train_df.drop(columns=["_patient_id"]),
        intraval_df.drop(columns=["_patient_id"]),
        len(train_patients),
        len(intraval_patients),
    )


def summarize_split(name, df, n_patients):
    labels = u_zeros_labels(df)
    positives = labels.sum(axis=0).astype(int)

    print(f"\n[{name}]")
    print(f"patients: {n_patients}")
    print(f"rows: {len(df)}")
    print("positive counts after U-Zeros:")
    for label in TARGET_LABELS:
        print(f"  {label}: {positives[label]}")


def main():
    parser = argparse.ArgumentParser(
        description="Split CheXpert train.csv into train/intraval CSVs by patient id."
    )
    parser.add_argument("--input", default="data/train.csv")
    parser.add_argument("--train-output", default="data/train_internal.csv")
    parser.add_argument("--intraval-output", default="data/intraval_internal.csv")
    parser.add_argument("--train-ratio", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    input_path = Path(args.input)
    train_output = Path(args.train_output)
    intraval_output = Path(args.intraval_output)

    df = pd.read_csv(input_path)
    if "Path" not in df.columns:
        raise ValueError("Input CSV must contain a 'Path' column.")

    missing_labels = [label for label in TARGET_LABELS if label not in df.columns]
    if missing_labels:
        raise ValueError(f"Missing target label columns: {missing_labels}")

    train_df, intraval_df, n_train_patients, n_intraval_patients = patientwise_split(
        df=df,
        train_ratio=args.train_ratio,
        seed=args.seed,
    )

    train_df.to_csv(train_output, index=False)
    intraval_df.to_csv(intraval_output, index=False)

    print(f"Saved train split to: {train_output}")
    print(f"Saved intra-validation split to: {intraval_output}")
    summarize_split("train", train_df, n_train_patients)
    summarize_split("intra-validation", intraval_df, n_intraval_patients)


if __name__ == "__main__":
    main()
