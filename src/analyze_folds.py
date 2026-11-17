
from pathlib import Path
from zipfile import ZipFile
from collections import defaultdict, Counter

import h5py
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "dataset" / "raw"

ARCHIVES = [
    "brainTumorDataPublic_1-766.zip",
    "brainTumorDataPublic_767-1532.zip",
    "brainTumorDataPublic_1533-2298.zip",
    "brainTumorDataPublic_2299-3064.zip",
]


def decode_pid(pid_data):
    values = pid_data.flatten()
    return "".join(
        chr(int(value))
        for value in values
        if int(value) != 0
    )


def main():
    records = []

    print("BUILDING DATASET FILE INDEX")
    print("=" * 55)

    for archive_name in ARCHIVES:
        archive_path = RAW_DIR / archive_name

        with ZipFile(archive_path, "r") as archive:
            mat_files = sorted(
                name for name in archive.namelist()
                if name.lower().endswith(".mat")
            )

            for filename in mat_files:
                with archive.open(filename) as file:
                    with h5py.File(file, "r") as data:
                        group = data["cjdata"]

                        records.append({
                            "file": filename,
                            "label": int(group["label"][0, 0]),
                            "pid": decode_pid(group["PID"][:]),
                        })

        print("Completed:", archive_name)

    with h5py.File(RAW_DIR / "cvind.mat", "r") as data:
        cvind = np.array(data["cvind"]).flatten().astype(int)

    print("\nINDEX VALIDATION")
    print("Dataset records:", len(records))
    print("CVIND entries:", len(cvind))

    if len(records) != len(cvind):
        print("ERROR: Record count and CVIND count do not match.")
        return

    print("\nFOLD DISTRIBUTION")
    fold_counts = Counter(cvind.tolist())

    for fold, count in sorted(fold_counts.items()):
        print(f"Fold {fold}: {count} images")

    print("\nCLASS DISTRIBUTION PER FOLD")

    for fold in sorted(fold_counts):
        class_counts = Counter(
            record["label"]
            for record, assigned_fold in zip(records, cvind)
            if assigned_fold == fold
        )

        print(f"\nFold {fold}:")
        for label in (1, 2, 3):
            print(f"  Label {label}: {class_counts[label]}")

    print("\nPATIENT OVERLAP ANALYSIS")

    patient_folds = defaultdict(set)

    for record, fold in zip(records, cvind):
        patient_folds[record["pid"]].add(int(fold))

    overlapping_patients = {
        pid: sorted(folds)
        for pid, folds in patient_folds.items()
        if len(folds) > 1
    }

    print("Unique patients:", len(patient_folds))
    print("Patients appearing in multiple folds:",
          len(overlapping_patients))

    if overlapping_patients:
        print("\nFirst 15 overlapping patient IDs:")
        for pid, folds in list(overlapping_patients.items())[:15]:
            print(f"PID {pid}: folds {folds}")

    print("\nAnalysis completed.")


if __name__ == "__main__":
    main()
