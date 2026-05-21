from __future__ import annotations

import argparse
import random
from pathlib import Path

import pandas as pd


def _nonempty(value) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    return text or None


def _group_id(row: pd.Series) -> str:
    isic_id = str(row["isic_id"])
    patient_id = _nonempty(row.get("patient_id"))
    lesion_id = _nonempty(row.get("lesion_id"))
    if patient_id:
        return f"patient:{patient_id}"
    if lesion_id:
        return f"lesion:{lesion_id}"
    return f"image:{isic_id}"


def prepare_isic_binary(
    metadata_csv: Path,
    image_dir: Path,
    output_dir: Path,
    val_frac: float,
    seed: int,
) -> None:
    metadata = pd.read_csv(metadata_csv)
    required = {"isic_id", "diagnosis_1"}
    missing = required - set(metadata.columns)
    if missing:
        raise ValueError(f"Missing required metadata columns: {sorted(missing)}")

    df = metadata[metadata["diagnosis_1"].isin(["Benign", "Malignant"])].copy()
    df["sample_id"] = df["isic_id"].astype(str)
    df["image_path"] = df["sample_id"].map(lambda x: str(image_dir / f"{x}.jpg"))
    df = df[df["image_path"].map(lambda p: Path(p).exists())].copy()
    if df.empty:
        raise ValueError(
            "No labelled ISIC images were found. Check --metadata_csv and --image_dir."
        )

    df["y"] = (df["diagnosis_1"] == "Malignant").astype(int)
    df["patient_global"] = df.apply(_group_id, axis=1)
    df["dataset_id"] = "ISIC"

    groups = df["patient_global"].drop_duplicates().tolist()
    rng = random.Random(seed)
    rng.shuffle(groups)
    n_val = max(1, int(round(len(groups) * val_frac)))
    val_groups = set(groups[:n_val])

    val_df = df[df["patient_global"].isin(val_groups)].copy()
    train_df = df[~df["patient_global"].isin(val_groups)].copy()

    columns = ["sample_id", "image_path", "y", "patient_global", "dataset_id"]
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "isic_binary_train.csv"
    val_path = output_dir / "isic_binary_val.csv"

    train_df[columns].to_csv(train_path, index=False)
    val_df[columns].to_csv(val_path, index=False)

    print(f"Wrote {train_path} ({len(train_df):,} rows)")
    print(train_df["y"].value_counts().sort_index().rename({0: "benign", 1: "malignant"}))
    print(f"\nWrote {val_path} ({len(val_df):,} rows)")
    print(val_df["y"].value_counts().sort_index().rename({0: "benign", 1: "malignant"}))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare ISIC benign/malignant CSVs for Stage 1 binary pretraining."
    )
    parser.add_argument("--metadata_csv", default="isic_images/metadata.csv")
    parser.add_argument("--image_dir", default="isic_images")
    parser.add_argument("--output_dir", default="Dataset/isic_binary")
    parser.add_argument("--val_frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    prepare_isic_binary(
        metadata_csv=Path(args.metadata_csv),
        image_dir=Path(args.image_dir),
        output_dir=Path(args.output_dir),
        val_frac=args.val_frac,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
