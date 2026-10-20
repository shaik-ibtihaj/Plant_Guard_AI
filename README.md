# Plant Guard AI 🌿

> Intelligent Plant Disease Detection & Severity Assessment powered by Deep Learning

[![CI Pipeline](https://github.com/your-username/Plant_Guard_AI/actions/workflows/ci.yml/badge.svg)](https://github.com/your-username/Plant_Guard_AI/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-18%2B-61DAFB.svg)](https://react.dev/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)

---

# 🌱 Overview

**Plant Guard AI** is an end-to-end AI-powered platform for:

- 🔍 Plant Disease Detection using Deep Learning
- 📊 Disease Severity Assessment
- 🧠 Explainability with Grad-CAM
- 💊 Treatment Recommendation System
- 🚀 FastAPI Backend API
- 🌐 React Frontend Application

The project combines multiple plant disease datasets, advanced computer vision models, explainable AI techniques, and a production-ready deployment pipeline.

---

# 🏗️ Architecture

```text
Plant Guard AI
│
├── AI Module
│   ├── Dataset Pipeline
│   ├── Model Training
│   ├── Evaluation
│   ├── Explainability
│   └── Severity Assessment
│
├── Backend API
│   └── FastAPI
│
├── Frontend
│   └── React + TypeScript
│
├── Deployment
│   └── Docker
│
└── Monitoring & CI/CD
```

---

# 🚀 Quick Start

## Prerequisites

- Python 3.11+
- Node.js 20+
- Docker & Docker Compose

## Clone Repository

```bash
git clone https://github.com/your-username/Plant_Guard_AI.git
cd Plant_Guard_AI
```

## Install Dependencies

```bash
pip install -r requirements.txt
pip install -r backend/requirements.txt
```

## Run Backend

```bash
uvicorn backend.app.main:app --reload
```

API documentation is available at `http://localhost:8000/docs`.

## Run Frontend

```bash
cd frontend
npm install
npm run dev
```

## Docker

```bash
cd docker
docker-compose up --build
```

---

# 📁 Project Structure

```text
PLANT_GUARD_AI/

├── ai/
├── backend/
├── frontend/
├── data/
│   ├── raw/
│   ├── intermediate/
│   ├── processed/
│   ├── metadata/
│   └── quarantine/
│
├── models/
├── reports/
├── scripts/
│   ├── preprocessing/
│   └── analysis/
│
├── notebooks/
├── tests/
├── docs/
├── docker/
└── .github/
```

---

# 🧠 Implemented Models

| Model | Architecture | Purpose | Code |
|---------|-------------|-------------|---:|
| CNN | Custom CNN | Baseline | ✅ |
| EfficientNet-B0 | Transfer Learning | Production candidate | ✅ |
| ResNet50 | Transfer Learning | Benchmark | ✅ |
| ViT-B/16 | Vision Transformer | Advanced benchmark | ✅ |

---

# 📊 Dataset Overview

The dataset was created by combining multiple plant disease datasets and standardizing them into a unified taxonomy.

### Sources

- PlantVillage
- PlantVillage Extra
- Additional Plant Disease Datasets

### Final Dataset

| Metric | Value |
|----------|---------:|
| Total Images | 208,686 |
| Total Classes | 64 |
| Corrupted Images | 0 |

---

# 📈 Dataset Snapshot

| Metric | Value |
|----------|---------:|
| Total Images | 208,686 |
| Total Classes | 64 |
| Largest Class | 11,564 |
| Smallest Class | 53 |
| Imbalance Ratio | 218.19x |
| Minority Classes | 22 |
| Majority Classes | 7 |
| Corrupted Images | 0 |

---

# ✅ Phase 0 — Project Bootstrap

Completed initial project architecture and repository setup.

Created:

```text
PLANT_GUARD_AI/

├── ai/
├── backend/
├── frontend/
├── data/
├── models/
├── reports/
├── scripts/
├── tests/
├── docs/
├── docker/
└── .github/
```

---

# ✅ Phase 1 — Dataset Collection & Analysis

## Datasets Collected

| Dataset | Images |
|----------|---------:|
| PlantVillage | 41,276 |
| PlantVillage Extra | 162,916 |
| Total | 204,192+ |

## Script Executed

```bash
python scripts/analyze_dataset.py
```

## Results

```text
Total Images       : 208,694
Corrupted Images   : 0
```

## Generated Files

```text
data/metadata/
├── dataset_summary.csv
└── corrupted_images.txt
```

---

# ✅ Phase 2.1 — Dataset Cleaning

## Purpose

- Validate every image
- Detect corrupted files
- Remove invalid files
- Quarantine problematic samples

## Script

```bash
python scripts/clean_dataset.py
```

## Results

```text
Total files scanned : 208,699
Valid images        : 208,694
Corrupted images    : 0
Invalid files       : 5
```

## Outputs

```text
data/metadata/
├── cleaning_report.csv
└── cleaning_summary.txt

data/quarantine/
└── 5 invalid files
```

---

# ✅ Phase 2.2 — Label Standardization

## Purpose

Convert all datasets into a unified naming convention.

### Example

```text
Mango healthy (P0a)
→ Mango___healthy

Mango diseased (P0b)
→ Mango___diseased

Tomato_Early_blight
→ Tomato___Early_blight
```

## Script

```bash
python scripts/standardize_labels.py
```

## Results

```text
Discovered Classes     : 211
Final Classes          : 64
Mappings Generated     : 72
```

## Outputs

```text
data/metadata/
└── class_mapping.json
```

---

# ✅ Phase 2.3 — Dataset Merge

## Purpose

Merge all standardized datasets into one master dataset.

## Script

```bash
python scripts/merge_datasets.py
```

## Results

```text
Classes Merged      : 210
Images Copied       : 208,686
Final Classes       : 64
```

## Outputs

```text
data/metadata/
└── merge_report.csv
```

---

# ✅ Phase 2.4 — Dataset Splitting

## Purpose

Create train, validation, and test sets.

## Split Strategy

```text
Train       80%
Validation  10%
Test        10%
```

## Script

```bash
python scripts/split_dataset.py
```

## Results

```text
64 Classes
208,686 Images
```

## Outputs

```text
data/processed/

├── train/
├── val/
└── test/

data/metadata/
├── split_report.csv
└── split_summary.txt
```

---

# ✅ Phase 3.1 — RGB Mean & Standard Deviation

## Purpose

Compute normalization values required for training.

Used later for:

- EfficientNet
- ResNet50
- Vision Transformer

## Output

```text
metadata/
└── rgb_stats.json
```

---

# ✅ Phase 3.2 — Class Distribution Analysis

## Purpose

Analyze dataset distribution across classes.

## Outputs

```text
metadata/
└── class_distribution.csv

reports/
└── class_distribution.png
```

## Results

```text
Total Classes : 64
Total Images  : 146,054
```

### Largest Classes

| Class | Images |
|---------|---------:|
| Orange___Haunglongbing_(Citrus_greening) | 11,564 |
| Tomato___Tomato_Yellow_Leaf_Curl_Virus | 11,249 |
| Soybean___healthy | 10,689 |

---

# ✅ Phase 3.3 — Class Imbalance Analysis

## Purpose

Identify minority and majority classes.

## Outputs

```text
metadata/
├── class_imbalance_report.json
├── majority_classes.csv
└── minority_classes.csv

reports/
└── class_imbalance_histogram.png
```

## Results

```text
Largest Class      : 11,564 images
Smallest Class     : 53 images
Imbalance Ratio    : 218.19x

Minority Classes   : 22
Majority Classes   : 7
```

### Largest Class

```text
Orange___Haunglongbing_(Citrus_greening)
```

### Smallest Class

```text
Lemon___diseased
```

---

# ✅ Phase 3.4 — Image Resolution Analysis

## Purpose

Analyze image dimensions and dataset consistency.

## Outputs

```text
metadata/
└── image_resolution_stats.json

reports/
├── width_distribution.png
├── height_distribution.png
└── resolution_scatter.png
```

## Results

### Width

```text
Min      : 256 px
Max      : 6000 px
Mean     : 379.70 px
Median   : 256 px
```

### Height

```text
Min      : 256 px
Max      : 4000 px
Mean     : 336.63 px
Median   : 256 px
```

### Aspect Ratio

```text
Mean     : 1.01
Median   : 1.00
```

### Resolution Findings

```text
Most images are 256×256
Images are predominantly square
Few high-resolution field images exist
Corrupted Images: 0
```

### Training Implications

The dataset is well suited for:

- EfficientNet-B0 (224×224)
- ResNet50 (224×224)
- ViT-B/16 (224×224)

after standard resizing.

---

# ✅ Phase 4 — Exploratory Data Analysis (EDA) & Training Preparation

## Purpose

Perform comprehensive exploratory data analysis on the prepared training dataset and generate actionable insights for model development.

---

## 4.1 Dataset Distribution Analysis

### Outputs

```text
datasets/metadata/
└── class_distribution.csv

reports/
└── class_distribution.png
```

### Results

```text
Total Classes : 64
Training Images : 146,054
```

### Key Findings

```text
Largest Class:
Orange___Haunglongbing_(Citrus_greening)
11,564 images

Smallest Class:
Lemon___diseased
53 images
```

---

## 4.2 Healthy vs Diseased Analysis

### Outputs

```text
datasets/metadata/
└── disease_health_summary.json

reports/
└── disease_vs_healthy.png
```

### Results

| Category | Classes | Images | Percentage |
|-----------|---------:|---------:|---------:|
| Healthy | 24 | 37,768 | 25.86% |
| Diseased | 40 | 108,286 | 74.14% |
| Total | 64 | 146,054 | 100% |

### Findings

```text
Disease samples dominate the dataset.

Healthy classes:
24

Disease classes:
40
```

---

## 4.3 Class Imbalance Analysis

### Outputs

```text
datasets/metadata/
├── class_imbalance_report.json
├── majority_classes.csv
└── minority_classes.csv

reports/
└── class_imbalance_histogram.png
```

### Results

| Metric | Value |
|----------|---------:|
| Largest Class | 11,564 |
| Smallest Class | 53 |
| Imbalance Ratio | 218.19x |
| Majority Classes | 7 |
| Minority Classes | 22 |

### Training Implications

```text
Severe class imbalance exists.

Recommended:
• WeightedRandomSampler
• Class-weighted loss
• Focal Loss
• Targeted augmentation
```

---

## 4.4 Image Resolution Analysis

### Outputs

```text
datasets/metadata/
└── image_resolution_stats.json

reports/
├── width_distribution.png
├── height_distribution.png
└── resolution_scatter.png
```

### Results

#### Width

```text
Min      : 256 px
Max      : 6000 px
Mean     : 379.70 px
Median   : 256 px
```

#### Height

```text
Min      : 256 px
Max      : 4000 px
Mean     : 336.63 px
Median   : 256 px
```

#### Aspect Ratio

```text
Mean     : 1.01
Median   : 1.00
```

### Findings

```text
Most images are 256×256.
Most images are nearly square.
Dataset is suitable for 224×224 model input resizing.
```

---

## 4.5 Sample Visualization

### Outputs

```text
reports/
└── sample_grid_overview.png
```

### Features

```text
25 randomly sampled images
5 × 5 visualization grid
Class labels displayed
Healthy and diseased samples highlighted
```

---

## 4.6 EDA Insights & Reporting

### Outputs

```text
datasets/metadata/
├── eda_insights.json

reports/
└── dataset_eda_report.md
```

### Generated Insights

```text
Total Images:
208,686

Total Classes:
64

Corrupted Images:
0

Largest Class:
11,564 images

Smallest Class:
53 images

Imbalance Ratio:
218.19x
```

### Training Recommendations

```text
Input Size:
224 × 224

Augmentation:
RandomFlip
Rotation
Affine
ColorJitter

Sampling:
WeightedRandomSampler

Loss:
Focal Loss (γ=2)

Models:
EfficientNet-B0
ResNet50
ViT-B/16
```

---

# 🏆 Phase 4 Validation Audit

A full validation audit was performed across all metadata, visualizations, and reports.

### Audit Summary

| Check | Status |
|---------|---------|
| File Integrity | ✅ |
| CSV Validation | ✅ |
| JSON Validation | ✅ |
| Numerical Consistency | ✅ |
| Visualization Review | ✅ |
| Report Review | ✅ |
| Cross-File Verification | ✅ |

### Audit Results

```text
Total Artifacts Reviewed : 15

PASS     : 15
WARNING  : 2
FAIL     : 0
```

### Final Score

```text
Phase 4 Score : 100 / 100
```

### Verdict

```text
READY FOR PHASE 5
```

---

# 🎯 Current Project Status

| Phase | Status |
|---------|---------|
| Phase 0 — Project Setup | ✅ |
| Phase 1 — Dataset Collection | ✅ |
| Phase 2.1 — Dataset Cleaning | ✅ |
| Phase 2.2 — Label Standardization | ✅ |
| Phase 2.3 — Dataset Merge | ✅ |
| Phase 2.4 — Dataset Splitting | ✅ |
| Phase 3.1 — RGB Mean & Std | ✅ |
| Phase 3.2 — Class Distribution | ✅ |
| Phase 3.3 — Class Imbalance Analysis | ✅ |
| Phase 3.4 — Image Resolution Statistics | ✅ |
| Phase 4.1 — Dataset Distribution Analysis | ✅ |
| Phase 4.2 — Healthy vs Diseased Analysis | ✅ |
| Phase 4.3 — Class Imbalance Visualization | ✅ |
| Phase 4.4 — Image Resolution Analysis | ✅ |
| Phase 4.5 — Sample Visualization | ✅ |
| Phase 4.6 — EDA Reporting & Insights | ✅ |
| Phase 5 — Model Development & Training | ✅ Completed (constrained CPU benchmark) |
| Phase 6 — Model Evaluation | ✅ Completed (constrained test evaluation) |
| Phase 7 — Explainable AI (Grad-CAM) | ✅ CNN and ViT explanations |
| Phase 8 — Severity Assessment | ✅ Mask-based area assessment |
| Phase 9 — Backend API | ✅ API integration complete |
| Phase 10 — Frontend | ✅ Responsive analysis dashboard and API integration |
| Phase 11 — Deployment & CI/CD | ⏳ |

---

# 🚀 Phase 5 — Model Development & Training

| Task | Implementation | Execution |
|---|---:|---:|
| 5.1 PyTorch Dataset Class | ✅ | ✅ Dataset verified |
| 5.2 DataLoader Pipeline | ✅ | ✅ Splits verified |
| 5.3 Class Weights | ✅ | ✅ 64 classes |
| 5.4 Data Augmentation | ✅ | ✅ Configured |
| 5.5 Baseline CNN | ✅ | ✅ CPU benchmark |
| 5.6 EfficientNet-B0 | ✅ | ✅ CPU benchmark |
| 5.7 ResNet50 | ✅ | ✅ CPU benchmark |
| 5.8 ViT-B/16 | ✅ | ✅ CPU benchmark |
| 5.9 Model Comparison | ✅ | ✅ Report generated |
| 5.10 Production Selection | ✅ | ✅ EfficientNet-B0 selected |

The data pipeline uses the RGB statistics in
`datasets/metadata/rgb_stats.json`, enforces one class mapping across all
splits, and supports reproducible inverse-frequency sampling. Validation and
test transforms are deterministic.

```python
from ai.data import create_data_pipeline

pipeline = create_data_pipeline()
images, labels = next(iter(pipeline.train))
```

All four architectures use the same training and evaluation pipeline. It
provides AdamW optimization, validation-based learning-rate scheduling, CUDA
mixed precision, early stopping, atomic checkpoint saving, and accuracy,
precision, recall, macro-F1, weighted-F1, and confusion-matrix evaluation.

Run each experiment from the repository root in a CUDA-enabled environment:

```bash
python -m ai.training.run_phase5 --epochs 30 --batch-size 16 --workers 4
```

The command above performs tasks 5.5–5.10 end to end. Models can also be run
individually when experiments need different batch sizes:

```bash
python -m ai.training.train --model baseline_cnn --epochs 30 --batch-size 32 --workers 4
python -m ai.training.train --model efficientnet_b0 --epochs 30 --batch-size 32 --workers 4
python -m ai.training.train --model resnet50 --epochs 30 --batch-size 32 --workers 4
python -m ai.training.train --model vit_b_16 --epochs 30 --batch-size 16 --workers 4
```

Compare the completed runs and select the production checkpoint:

```bash
python -m ai.training.compare models/runs/baseline_cnn_metrics.json models/runs/efficientnet_b0_metrics.json models/runs/resnet50_metrics.json models/runs/vit_b_16_metrics.json
```

The comparison ranks models by validation macro-F1, then weighted-F1,
accuracy, and loss. It generates `reports/model_comparison.json`,
`reports/model_comparison.md`, `models/production_model.json`, and copies the
winner to `models/best_model.pth`. Checkpoints and per-run histories remain
ignored by Git because they are generated artifacts.

### Phase 5 Benchmark Results

The reproducible CPU benchmark was executed on October 20, 2026 with seed 42,
PyTorch 2.13 CPU, 224×224 inputs, one epoch, four training images per class
(256 total), two validation images per class (128 total), weighted sampling,
ImageNet-pretrained transfer models, and frozen transfer backbones.

| Rank | Model | Accuracy | Macro Precision | Macro Recall | Macro F1 | Loss |
|---:|---|---:|---:|---:|---:|---:|
| 1 | EfficientNet-B0 | 10.94% | 3.20% | 10.94% | 4.74% | 4.0638 |
| 2 | ResNet50 | 4.69% | 2.07% | 4.69% | 1.94% | 4.0826 |
| 3 | ViT-B/16 | 1.56% | 0.95% | 1.56% | 1.07% | 4.2351 |
| 4 | Baseline CNN | 3.12% | 0.41% | 3.12% | 0.64% | 4.8331 |

**Selected model:** EfficientNet-B0, ranked by validation macro-F1. The
checkpoint is stored at `models/best_model.pth`, its manifest at
`models/production_model.json`, and the complete ranking at
`reports/model_comparison.md`.

> **Scope warning:** Phase 5 is complete as a constrained benchmark, but the
> selected checkpoint is not ready for real-world deployment. Its 10.94%
> validation accuracy reflects the deliberately small CPU run. A CUDA-backed
> full-dataset run using the 30-epoch command above is required before release.

### Phase 5 Goals

- Build a robust PyTorch training pipeline
- Address severe class imbalance (218.19x)
- Train multiple deep learning architectures
- Compare performance using Accuracy, Precision, Recall, and F1-Score
- Select the best-performing production model
- Prepare for Explainable AI integration (Grad-CAM)

---

# 📊 Phase 6 — Model Evaluation

## Completion Status

| Task | Status |
|---|---:|
| 6.1 Evaluate selected checkpoint on held-out test data | ✅ |
| 6.2 Compute top-1/top-5 and aggregate metrics | ✅ |
| 6.3 Generate per-class metrics and confusion matrices | ✅ |
| 6.4 Export individual predictions and error analysis | ✅ |
| 6.5 Generate calibration and misclassification diagnostics | ✅ |
| 6.6 Produce the final evaluation report | ✅ |

The selected EfficientNet-B0 checkpoint was evaluated on a deterministic,
class-balanced test subset containing two images from each of the 64 classes
(128 images total), matching the constrained Phase 5 validation scope.

| Metric | Result |
|---|---:|
| Test loss | 4.0514 |
| Top-1 accuracy | 8.59% |
| Top-5 accuracy | 25.00% |
| Macro F1 | 5.28% |
| Weighted F1 | 5.28% |
| Mean confidence | 2.91% |
| Expected calibration error (ECE) | 5.68% |
| Misclassifications | 117 / 128 (91.41%) |

Generated artifacts:

- `reports/phase6/test_metrics.json`
- `reports/phase6/per_class_metrics.csv`
- `reports/phase6/confusion_matrix.png`
- `reports/phase6/confusion_matrix_normalized.png`
- `reports/phase6/predictions.csv`
- `reports/phase6/error_analysis.json`
- `reports/phase6/reliability_diagram.png`
- `reports/phase6/misclassification_gallery.png`
- `reports/phase6/evaluation_report.md`

The evaluation command is reproducible and supports either a deterministic
per-class subset or the entire split. Run it without
`--eval-samples-per-class` after full GPU training to satisfy the production
release gate:

```bash
python -m ai.training.evaluate --model efficientnet_b0 --checkpoint models/best_model.pth --output reports/phase6/test_metrics_full.json --split test --batch-size 32 --workers 4 --artifacts-dir reports/phase6/full
```

> Phase 6 is complete for the constrained CPU benchmark scope. These results
> validate the evaluation workflow, not production model quality. The
> one-epoch Phase 5 checkpoint is not deployable; full training and complete
> test-split certification remain mandatory release gates.

---

# 🔎 Phase 7 — Explainable AI (Grad-CAM)

Phase 7 implements class-specific Grad-CAM overlays for the baseline CNN,
EfficientNet-B0, and ResNet50, plus gradient-weighted patch-token attribution
for ViT-B/16. The command reports the predicted class and confidence, saves an
RGB overlay, and writes a JSON sidecar with the explained class and input
paths. By default, it explains the predicted class; pass `--target-class` to
explain a chosen class index.

```bash
python -m ai.explainability.gradcam --model efficientnet_b0 --checkpoint models/best_model.pth --image datasets/processed/test/Alstonia_Scholaris___diseased/0014_0002.JPG --output reports/phase7/gradcam_overlay.png
```

The `GradCAM` and `TransformerTokenCAM` classes can also be used directly.
`get_target_layer(model, model_name)` selects the final convolutional block or
ViT encoder block as appropriate. The transformer heatmap uses class-token
gradients projected onto the square patch grid; it is a gradient attribution
map and does not represent a segmented disease region.

The example output is saved as `reports/phase7/gradcam_overlay.png`, with
metadata at `reports/phase7/gradcam_overlay.json`. This example uses the
center crop presented to the model. It uses the constrained one-epoch
checkpoint and is a pipeline demonstration. Its heatmap is diffuse and
includes substantial background response, so it should not be interpreted as
a reliable disease localization until the model is fully trained and the
explanation is reviewed on representative images.

---

# 🌿 Phase 8 — Severity Assessment

Phase 8 provides an end-to-end mask-based area assessment. It measures lesion
pixels inside a leaf mask, maps the affected-area ratio to Healthy, Mild,
Moderate, Severe, or Critical, highlights the measured region, and writes a
JSON report with the measurements and conservative recommendation. The
classifier confidence is recorded but does not determine severity. Without a
measured area, `SeverityAssessor` returns Unknown.

```python
from ai.severity import SeverityAssessor

assessor = SeverityAssessor()
result = assessor.assess_with_details(
    prediction_confidence=0.91,
    affected_area_ratio=0.24,
)
```

For image assessment, provide pixel-aligned grayscale masks. Any non-black
pixel is treated as foreground:

```bash
python -m ai.severity.assessment --image leaf.jpg --leaf-mask leaf_mask.png --lesion-mask lesion_mask.png --prediction-confidence 0.91 --output reports/phase8/severity_overlay.png
```

The overlay highlights lesion pixels that overlap the leaf mask; the JSON
sidecar records leaf and affected pixel counts, affected-area ratio, severity,
and recommendation. Default boundaries are 5%, 20%, and 40%: zero is
Healthy; up to 5% is Mild; over 5% through 20% is Moderate; over 20% through
40% is Severe; and over 40% is Critical. Thresholds are configurable through
`SeverityAssessor(area_thresholds=...)`.

The project does not yet generate leaf or lesion masks automatically. The
assessment is therefore auditable when masks are supplied, but depends on
their quality. Thresholds are initial heuristics and have not been validated
for field use; recommendations do not prescribe pesticides or replace expert
agronomic advice.

---

# 🚀 Phase 9 — Backend API

Phase 9 is complete. The FastAPI service loads the selected checkpoint once at
startup and exposes image classification through `POST /api/v1/predict`.
Upload a JPEG, PNG, or WEBP image as multipart field `file`; optional `top_k`
is limited to 1–10. Responses include the predicted class, confidence, top
classes, image dimensions, checkpoint epoch, and inference time. Uploaded
images are processed in memory and are not saved by the API.

```bash
uvicorn backend.app.main:app --reload
```

- `GET /health` is the process liveness check.
- `GET /health/ready` and `GET /api/v1/health` report model and database readiness.
- `POST /api/v1/predict` performs image classification.
- `POST /api/v1/explain` returns a Grad-CAM/ViT attribution overlay as base64 PNG.
- `POST /api/v1/severity` assesses supplied, image-aligned leaf and lesion masks.
- `GET /api/v1/records` lists metadata history; `DELETE /api/v1/records/{record_id}` removes a record.
- `GET /docs` provides interactive OpenAPI documentation.

Configure the model with `PLANT_GUARD_MODEL` and
`PLANT_GUARD_CHECKPOINT`. Optional path settings include
`PLANT_GUARD_CLASS_MAPPING` and `PLANT_GUARD_RGB_STATS`. `DATABASE_URL` selects
the database (local default: SQLite at `data/plant_guard.db`; Compose uses
PostgreSQL). Set `PLANT_GUARD_API_KEY` to enable `X-API-Key` authentication;
production requires at least 32 characters. `PLANT_GUARD_RATE_LIMIT_PER_MINUTE`,
`PLANT_GUARD_IMAGE_SIZE`,
`PLANT_GUARD_MAX_UPLOAD_BYTES`, and comma-separated
`PLANT_GUARD_CORS_ORIGINS` adjust preprocessing and service limits. Paths may
be absolute or relative to the repository root. A missing or invalid model
keeps the liveness endpoint available while readiness and analysis return
HTTP 503. The fixed-window limiter is in-memory and per process; use a shared
gateway or store for multi-worker or replicated deployments.

The container can be launched from the `docker` directory with
`docker compose up --build`. The backend image is built from the repository
root so it includes the AI package, model checkpoint, class mapping, and RGB
statistics.

Phase 9 is complete as an API/integration milestone, not as a production
diagnostic release. Severity depends on supplied masks; authentication is a
shared API key rather than user/role management, and the rate limiter is
in-memory per process. The checked-in benchmark checkpoint has very low
accuracy. Full training and field validation remain release gates; outputs are
advisory and not a diagnosis.

---

# 🌱 Phase 10 — Frontend

The React + TypeScript frontend provides a responsive workspace for plant
diagnosis, mask-based severity measurement, and metadata history. Diagnosis
supports JPEG/PNG/WEBP uploads up to 10 MB, top-five predictions, confidence,
and on-demand Grad-CAM/ViT overlays. Severity measurement accepts an image
with pixel-aligned leaf and lesion masks. History can be refreshed and records
deleted; the original uploaded images are never retained by the API.

Run the frontend from the `frontend` directory with `npm install` and
`npm run dev`. Vite proxies API and health requests to `http://localhost:8000`.
In Docker Compose, the included Nginx configuration serves the app and proxies
API/health requests to the backend service. Optionally provide an API key in
the in-app API access setting; it is stored in browser local storage and sent
as `X-API-Key`.

The UI presents model results as advisory estimates, not diagnoses. Severity
depends on user-supplied masks; this frontend does not generate masks or
provide treatment prescriptions.
