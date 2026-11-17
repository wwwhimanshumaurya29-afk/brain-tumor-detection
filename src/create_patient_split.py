
from pathlib import Path
from zipfile import ZipFile
from collections import defaultdict, Counter

import h5py
import numpy as np
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "dataset" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "dataset" / "processed"

ARCHIVES = [
    "brainTumorDataPublic_1-766.zip",
    "brainTumorDataPublic_767-1532.zip",
    "brainTumorDataPublic_1533-2298.zip",
    "brainTumorDataPublic_2299-3064.zip",
]

RANDOM_SEED = 42
TEST_SIZE = 0.20


def decode_pid(pid_data):
    values = pid_data.flatten()
    return "".join(
        chr(int(value))
        for value in values
        if int(value) != 0
    )


def collect_records():
    patients = defaultdict(list)

    for archive_name in ARCHIVES:
        archive_path = RAW_DIR / archive_name

        with ZipFile(archive_path, "r") as archive:
            filenames = sorted(
                name for name in archive.namelist()
                if name.lower().endswith(".mat")
            )

            for filename in filenames:
                with archive.open(filename) as file:
                    with h5py.File(file, "r") as data:
                        group = data["cjdata"]

                        pid = decode_pid(group["PID"][:])
                        label = int(group["label"][0, 0])

                        patients[pid].append({
                            "archive": archive_name,
                            "file": filename,
                            "label": label,
                        })

    return patients


def main():
    print("PATIENT-LEVEL TRAIN/TEST SPLIT")
    print("=" * 55)

    patients = collect_records()

    patient_ids = sorted(patients)

    patient_labels = []

    for pid in patient_ids:
        labels = {
            record["label"]
            for record in patients[pid]
        }

        if len(labels) != 1:
            raise ValueError(
                f"Patient {pid} has inconsistent labels: {labels}"
            )

        patient_labels.append(next(iter(labels)))

    train_ids, test_ids = train_test_split(
        patient_ids,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=patient_labels,
    )

    train_set = set(train_ids)
    test_set = set(test_ids)

    # Verify patient separation
    assert train_set.isdisjoint(test_set)

    train_records = [
        {
            **record,
            "pid": pid
        }
        for pid in train_ids
        for record in patients[pid]
    ]

    test_records = [
        {
            **record,
            "pid": pid
        }
        for pid in test_ids
        for record in patients[pid]
    ]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    def save_split(records, filename):
        output_path = OUTPUT_DIR / filename

        with output_path.open("w", encoding="utf-8") as file:
            file.write("archive,filename,pid,label\n")

            for record in records:
                file.write(
                    f"{record['archive']},"
                    f"{record['file']},"
                    f"{record['pid']},"
                    f"{record['label']}\n"
                )

        print(f"Saved: {output_path}")

    save_split(train_records, "train.csv")
    save_split(test_records, "test.csv")

    print("\nSPLIT SUMMARY")
    print("Training patients:", len(train_ids))
    print("Testing patients:", len(test_ids))
    print("Training images:", len(train_records))
    print("Testing images:", len(test_records))
    print("Patient overlap:", len(train_set & test_set))

    print("\nTRAINING CLASS DISTRIBUTION")
    print(Counter(r["label"] for r in train_records))

    print("\nTEST CLASS DISTRIBUTION")
    print(Counter(r["label"] for r in test_records))

    print("\nSplit creation completed.")


if __name__ == "__main__":
    main()
