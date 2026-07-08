from pathlib import Path
from PIL import Image
import pandas as pd

# ==========================================================
# CONFIG
# ==========================================================

DATASET_ROOT = Path("datasets/raw")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# ==========================================================
# ANALYSIS
# ==========================================================

results = []
corrupted_files = []

print("\n" + "=" * 60)
print("PLANTGUARD AI - DATASET ANALYSIS")
print("=" * 60)

for dataset in DATASET_ROOT.iterdir():

    if not dataset.is_dir():
        continue

    print(f"\nDataset: {dataset.name}")
    print("-" * 60)

    dataset_total = 0
    class_count = 0

    for class_dir in dataset.iterdir():

        if not class_dir.is_dir():
            continue

        class_count += 1

        images = []

        for ext in IMAGE_EXTENSIONS:
            images.extend(class_dir.rglob(f"*{ext}"))

        image_count = len(images)
        dataset_total += image_count

        print(f"{class_dir.name:<40} {image_count:>6}")

        # Check for corrupted images
        for img_path in images:
            try:
                img = Image.open(img_path)
                img.verify()
            except Exception:
                corrupted_files.append(str(img_path))

        results.append(
            {
                "dataset": dataset.name,
                "class": class_dir.name,
                "images": image_count,
            }
        )

    print("-" * 60)
    print(f"Classes: {class_count}")
    print(f"Images : {dataset_total}")

# ==========================================================
# SAVE REPORTS
# ==========================================================

metadata_dir = Path("datasets/metadata")
metadata_dir.mkdir(parents=True, exist_ok=True)

df = pd.DataFrame(results)

df.to_csv(
    metadata_dir / "dataset_summary.csv",
    index=False
)

with open(metadata_dir / "corrupted_images.txt", "w") as f:
    for file in corrupted_files:
        f.write(file + "\n")

# ==========================================================
# FINAL SUMMARY
# ==========================================================

print("\n" + "=" * 60)
print("FINAL SUMMARY")
print("=" * 60)

print(f"Total Datasets       : {df['dataset'].nunique()}")
print(f"Total Classes        : {len(df)}")
print(f"Total Images         : {df['images'].sum()}")
print(f"Corrupted Images     : {len(corrupted_files)}")

print("\nReports Generated:")
print("✓ datasets/metadata/dataset_summary.csv")
print("✓ datasets/metadata/corrupted_images.txt")

print("\nAnalysis Complete!")