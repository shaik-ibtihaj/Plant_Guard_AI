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
```

## Run Backend

```bash
cd backend
uvicorn app.main:app --reload
```

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

# 🧠 Planned Models

| Model | Architecture | Purpose |
|---------|-------------|-------------|
| CNN | Custom CNN | Baseline |
| EfficientNet-B0 | Transfer Learning | Primary Model |
| ResNet50 | Transfer Learning | Benchmark |
| ViT-B/16 | Vision Transformer | Advanced Benchmark |

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

# 📂 Current Dataset Pipeline

```text
data/

├── raw/
│
├── intermediate/
│
├── processed/
│   ├── train/
│   ├── val/
│   └── test/
│
├── quarantine/
│
└── metadata/
    ├── class_mapping.json
    ├── cleaning_report.csv
    ├── merge_report.csv
    ├── split_report.csv
    ├── rgb_stats.json
    ├── class_distribution.csv
    ├── class_imbalance_report.json
    └── image_resolution_stats.json
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
| Phase 4 — Training Pipeline Preparation | ⏳ |
| Phase 5 — Model Training | ⏳ |
| Phase 6 — Evaluation | ⏳ |
| Phase 7 — Explainability (Grad-CAM) | ⏳ |
| Phase 8 — Severity Assessment | ⏳ |
| Phase 9 — Backend API | ⏳ |
| Phase 10 — Frontend | ⏳ |
| Phase 11 — Deployment & CI/CD | ⏳ |

---

# 🚀 Next Phase

## Phase 4 — Training Pipeline Preparation

Upcoming tasks:

- Compute Class Weights
- Create Dataset Metadata
- Build PyTorch Dataset Class
- Build DataLoader Pipeline
- Implement Data Augmentation
- Verify Train/Validation/Test Integrity
- Prepare Training Configuration

---

# 🤝 Contributing

Contributions, issues, and feature requests are welcome.

---

# 📄 License

This project is licensed under the MIT License.

See the LICENSE file for details.