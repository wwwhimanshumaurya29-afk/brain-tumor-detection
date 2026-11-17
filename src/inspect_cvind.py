

from pathlib import Path
from collections import Counter

import h5py
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MAT_FILE = PROJECT_ROOT / "dataset" / "raw" / "cvind.mat"


def main():
    print("CVIND VALUE ANALYSIS")
    print("=" * 50)

    if not MAT_FILE.exists():
        print("ERROR: cvind.mat not found:", MAT_FILE)
        return

    with h5py.File(MAT_FILE, "r") as data:
        cvind = np.array(data["cvind"]).flatten()

    print("Total entries:", len(cvind))
    print("Data type:", cvind.dtype)
    print("Minimum value:", np.min(cvind))
    print("Maximum value:", np.max(cvind))
    print("Contains NaN:", np.isnan(cvind).any())

    print("\nFirst 30 values:")
    print(cvind[:30])

    print("\nUnique values and their counts:")
    counts = Counter(cvind.tolist())

    for value, count in sorted(counts.items()):
        print(f"Value {value}: {count} entries")

    print("\nAnalysis completed.")


if __name__ == "__main__":
    main()
