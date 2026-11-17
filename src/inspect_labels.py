
from pathlib import Path
from zipfile import ZipFile
from collections import Counter

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
    """Decode MATLAB uint16 PID values into a readable string."""
    values = pid_data.flatten()

    try:
        return "".join(
            chr(int(value))
            for value in values
            if int(value) != 0
        )
    except (ValueError, TypeError):
        return str(values.tolist())


def main():
    label_counts = Counter()
    patient_ids = set()
    image_shapes = Counter()
    invalid_files = []
    total_mat_files = 0
    valid_files = 0

    print("BRAIN TUMOR DATASET ANALYSIS")
    print("=" * 60)

    for archive_name in ARCHIVES:
        archive_path = RAW_DIR / archive_name

        if not archive_path.exists():
            print(f"\nERROR: Archive not found: {archive_name}")
            continue

        print(f"\nInspecting: {archive_name}")

        with ZipFile(archive_path, "r") as archive:
            mat_files = [
                name for name in archive.namelist()
                if name.lower().endswith(".mat")
            ]

            for filename in mat_files:
                total_mat_files += 1

                try:
                    with archive.open(filename) as file:
                        with h5py.File(file, "r") as data:

                            if "cjdata" not in data:
                                raise ValueError(
                                    "Missing cjdata group"
                                )

                            group = data["cjdata"]

                            required_fields = [
                                "image",
                                "tumorMask",
                                "label",
                                "PID"
                            ]

                            for field in required_fields:
                                if field not in group:
                                    raise ValueError(
                                        f"Missing field: {field}"
                                    )

                            image = group["image"]
                            mask = group["tumorMask"]

                            image_shape = image.shape
                            mask_shape = mask.shape

                            # Check image dimensions
                            if len(image_shape) != 2:
                                raise ValueError(
                                    f"Image is not 2D: {image_shape}"
                                )

                            # Check mask dimensions
                            if len(mask_shape) != 2:
                                raise ValueError(
                                    f"Mask is not 2D: {mask_shape}"
                                )

                            # Check image-mask compatibility
                            if image_shape != mask_shape:
                                raise ValueError(
                                    f"Image shape {image_shape} "
                                    f"does not match mask shape {mask_shape}"
                                )

                            label = float(
                                group["label"][0, 0]
                            )

                            pid = decode_pid(
                                group["PID"][:]
                            )

                            if label not in (1.0, 2.0, 3.0):
                                raise ValueError(
                                    f"Unexpected class label: {label}"
                                )

                            label_counts[label] += 1
                            patient_ids.add(pid)
                            image_shapes[image_shape] += 1
                            valid_files += 1

                except Exception as error:
                    invalid_files.append(
                        (archive_name, filename, str(error))
                    )

        print(f"Completed: {archive_name}")
        print(f"MAT files found: {len(mat_files)}")

    # Final report
    print("\n" + "=" * 60)
    print("FINAL DATASET REPORT")
    print("=" * 60)

    print("\nFILE VALIDATION")
    print("Total MAT files found:", total_mat_files)
    print("Valid files:", valid_files)
    print("Invalid files:", len(invalid_files))

    print("\nCLASS DISTRIBUTION")
    for label, count in sorted(label_counts.items()):
        print(f"Label {int(label)}: {count} samples")

    print("Total valid labeled samples:", sum(label_counts.values()))

    print("\nIMAGE DIMENSIONS")
    for shape, count in sorted(image_shapes.items()):
        print(f"{shape}: {count} images")

    print("\nPATIENT ID ANALYSIS")
    print("Unique decoded PID values:", len(patient_ids))

    if patient_ids:
        print("Sample decoded PIDs:")
        for pid in sorted(patient_ids)[:10]:
            print(pid)

    if invalid_files:
        print("\nINVALID FILE DETAILS")
        for archive_name, filename, error in invalid_files:
            print(f"{archive_name} | {filename} | {error}")

    print("\nAnalysis completed.")


if __name__ == "__main__":
    main()
