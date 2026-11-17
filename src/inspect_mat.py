
from pathlib import Path
from zipfile import ZipFile
import h5py

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "dataset" / "raw"

ZIP_FILE = RAW_DIR / "brainTumorDataPublic_1-766.zip"
SAMPLE_FILE = "1.mat"


def inspect_hdf5(group, indent=0):
    for key in group.keys():
        item = group[key]
        prefix = " " * indent

        if isinstance(item, h5py.Dataset):
            print(f"{prefix}Variable: {key}")
            print(f"{prefix}Shape: {item.shape}")
            print(f"{prefix}Data type: {item.dtype}")

        elif isinstance(item, h5py.Group):
            print(f"{prefix}Group: {key}")
            inspect_hdf5(item, indent + 4)


def main():
    print("MATLAB v7.3 Dataset Inspection")
    print("-" * 45)

    with ZipFile(ZIP_FILE, "r") as archive:
        with archive.open(SAMPLE_FILE) as file:
            with h5py.File(file, "r") as data:
                print("\nVariables found:")
                inspect_hdf5(data)

    print("\nInspection completed.")


if __name__ == "__main__":
    main()
