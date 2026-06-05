"""Offline CSV preparation: patient-wise splitting and U-Zeros relabeling.

Kept separate from :mod:`chexpert.data` (runtime dataset) to mirror the
torchxrayvision convention of a dedicated data-preparation step.
"""

import re
from pathlib import Path

import pandas as pd

from chexpert.data import TARGET_LABELS


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


def split_patientwise(input_path, train_output, intraval_output, train_ratio, seed):
    input_path = Path(input_path)
    train_output = Path(train_output)
    intraval_output = Path(intraval_output)
    df = pd.read_csv(input_path)
    if "Path" not in df.columns:
        raise ValueError("Input CSV must contain a 'Path' column.")

    missing_labels = [label for label in TARGET_LABELS if label not in df.columns]
    if missing_labels:
        raise ValueError(f"Missing target label columns: {missing_labels}")

    train_df, intraval_df, n_train_patients, n_intraval_patients = patientwise_split(
        df=df,
        train_ratio=train_ratio,
        seed=seed,
    )

    train_df.to_csv(train_output, index=False)
    intraval_df.to_csv(intraval_output, index=False)

    print(f"Saved train split to: {train_output}")
    print(f"Saved intra-validation split to: {intraval_output}")
    summarize_split("train", train_df, n_train_patients)
    summarize_split("intra-validation", intraval_df, n_intraval_patients)


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


def preprocess_splits(splits, output_dir):
    output_dir = Path(output_dir)
    for split_path in splits:
        split_path = Path(split_path)
        output_path = output_dir / f"{split_path.stem}_uzeros.csv"
        preprocess_csv(split_path, output_path)
