import pandas as pd
from pathlib import Path
from src.data.preprocessing import preprocess_image

ROOT = Path(__file__).resolve().parents[1]

csv_path = ROOT / "dataset" / "processed" / "train_model.csv"
raw_dir = ROOT / "dataset" / "raw"

output_dir = ROOT / "sample_images"
output_dir.mkdir(exist_ok=True)

df = pd.read_csv(csv_path)
row = df.iloc[0]

image = preprocess_image(
    raw_dir / row["archive"],
    row["filename"],
    image_size=224
)

output_path = output_dir / "sample_mri.png"
image.save(output_path)

print("Sample MRI exported successfully!")
print("Image path:", output_path)
print("Original class label:", row["label"])