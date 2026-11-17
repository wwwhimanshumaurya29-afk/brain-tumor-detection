
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DIR = PROJECT_ROOT / "dataset" / "processed"

train_csv = PROCESSED_DIR / "train.csv"

df = pd.read_csv(train_csv)

# Each patient must have only one class label
patient_labels = df.groupby("pid")["label"].nunique()

if (patient_labels > 1).any():
    raise ValueError("Some patients have multiple class labels.")

# Create one row per patient
patients = (
    df.groupby("pid", as_index=False)["label"]
    .first()
)

# Split patients, not individual images
train_patients, val_patients = train_test_split(
    patients,
    test_size=0.20,
    random_state=42,
    stratify=patients["label"]
)

train_ids = set(train_patients["pid"])
val_ids = set(val_patients["pid"])

# Create image-level CSVs using patient IDs
train_df = df[df["pid"].isin(train_ids)].copy()
val_df = df[df["pid"].isin(val_ids)].copy()

# Verify no patient overlap
assert train_ids.isdisjoint(val_ids)

# Save new CSV files
train_df.to_csv(
    PROCESSED_DIR / "train_model.csv",
    index=False
)

val_df.to_csv(
    PROCESSED_DIR / "val.csv",
    index=False
)

print("PATIENT-LEVEL VALIDATION SPLIT")
print("==============================")

print("Total original training images:", len(df))
print("Total patients:", len(patients))

print("\nTraining images:", len(train_df))
print("Training patients:", len(train_ids))

print("\nValidation images:", len(val_df))
print("Validation patients:", len(val_ids))

print("\nTraining class distribution:")
print(train_df["label"].value_counts().sort_index())

print("\nValidation class distribution:")
print(val_df["label"].value_counts().sort_index())

print("\nPatient overlap:", len(train_ids & val_ids))

print("\nFiles saved successfully!")
