
from pathlib import Path
from zipfile import ZipFile, BadZipFile
from collections import Counter
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "dataset" / "raw"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
ARCHIVE_EXTENSIONS = {".zip"}

def inspect_archive(zip_path):
    print(f"\n{'=' * 65}")
    print(f"ARCHIVE: {zip_path.name}")
    print(f"{'=' * 65}")

    try:
        with ZipFile(zip_path, "r") as archive:
            files = [
                item for item in archive.infolist()
                if not item.is_dir()
            ]

            extensions = Counter(
                Path(item.filename).suffix.lower() or "[no extension]"
                for item in files
            )

            print(f"Total files: {len(files)}")
            print("\nFile types:")

            for extension, count in extensions.most_common():
                print(f"  {extension}: {count}")

            image_count = sum(
                count for ext, count in extensions.items()
                if ext in IMAGE_EXTENSIONS
            )
            print(f"\nImage files: {image_count}")

            print("\nFirst 15 archive entries:")
            for item in files[:15]:
                print(f"  {item.filename}")

            if len(files) > 15:
                print(f"  ... and {len(files) - 15} more files")

    except BadZipFile:
        print("ERROR: This ZIP file is incomplete or invalid.")
    except Exception as error:
        print(f"ERROR: {error}")


def inspect_readme(readme_path):
    print(f"\n{'=' * 65}")
    print(f"README: {readme_path.name}")
    print(f"{'=' * 65}")

    try:
        content = readme_path.read_text(
            encoding="utf-8",
            errors="replace"
        )
        print(content[:10000])

        if len(content) > 10000:
            print("\n[README output shortened]")

    except Exception as error:
        print(f"Could not read README: {error}")


def main():
    folder = Path(sys.argv[1]) if len(sys.argv) > 1 else DATASET_DIR

    if not folder.exists():
        print(f"Dataset folder not found: {folder}")
        print("Create dataset/raw and place downloaded files there.")
        return

    print("BRAIN TUMOR DATASET INSPECTION")
    print(f"Folder: {folder.resolve()}")

    files = [item for item in folder.iterdir() if item.is_file()]

    if not files:
        print("No files found in this folder yet.")
        return

    print(f"\nTotal files found: {len(files)}")

    for file in files:
        size_mb = file.stat().st_size / (1024 * 1024)
        print(f"  {file.name} ({size_mb:.2f} MB)")

    for file in files:
        if file.suffix.lower() in ARCHIVE_EXTENSIONS:
            inspect_archive(file)

        elif file.suffix.lower() in {".txt", ".md"}:
            inspect_readme(file)

    print("\nInspection completed.")
    print("No files were extracted, modified or deleted.")


if __name__ == "__main__":
    main()
