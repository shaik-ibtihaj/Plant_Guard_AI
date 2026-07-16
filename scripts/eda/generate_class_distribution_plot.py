import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# -------------------------
# Paths
# -------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]

CSV_PATH = PROJECT_ROOT / "datasets" / "metadata" / "class_distribution.csv"
OUTPUT_PATH = PROJECT_ROOT / "reports" / "class_distribution.png"

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# -------------------------
# Load Data
# -------------------------
df = pd.read_csv(CSV_PATH)

# Sort descending
df = df.sort_values("count", ascending=False)

# -------------------------
# Plot
# -------------------------
plt.figure(figsize=(14, 18))

plt.barh(
    df["class"],
    df["count"]
)

plt.gca().invert_yaxis()

plt.xlabel("Number of Images")
plt.ylabel("Class")
plt.title("PlantGuard AI - Class Distribution")

plt.tight_layout()

plt.savefig(
    OUTPUT_PATH,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(f"Saved plot to: {OUTPUT_PATH}")