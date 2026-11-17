
from pathlib import Path
from zipfile import ZipFile
from collections import defaultdict, Counter

import h5py

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
    patient_records = defaultdict(list)

    print("PATIENT-LEVEL DATASET ANALYSIS")
    print("=" * 55)

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

                        patient_records[pid].append({
                            "file": filename,
                            "label": label
                        })

    print("Unique patients:", len(patient_records))

    patient_image_counts = Counter(
        len(records)
        for records in patient_records.values()
    )

    print("\nImages per patient:")
    for count, patients in sorted(patient_image_counts.items()):
        print(f"{count} images: {patients} patients")

    print("\nPatient class consistency:")
    inconsistent = 0

    for pid, records in patient_records.items():
        labels = {record["label"] for record in records}

        if len(labels) > 1:
            inconsistent += 1
            print(f"PID {pid}: multiple labels {sorted(labels)}")

    print("Patients with multiple class labels:", inconsistent)

    print("\nLargest patient groups:")
    largest = sorted(
        patient_records.items(),
        key=lambda item: len(item[1]),
        reverse=True
    )[:15]

    for pid, records in largest:
        print(f"PID {pid}: {len(records)} images")

    print("\nAnalysis completed.")


if __name__ == "__main__":
    main()
