# Plant Guard AI 🌿

> Intelligent Plant Disease Detection & Severity Assessment powered by Deep Learning

[![CI Pipeline](https://github.com/your-username/Plant_Guard_AI/actions/workflows/ci.yml/badge.svg)](https://github.com/your-username/Plant_Guard_AI/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-18%2B-61DAFB.svg)](https://react.dev/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)

---

## 🌱 Overview

**Plant Guard AI** is an end-to-end AI-powered platform for:
- 🔍 **Plant Disease Detection** using state-of-the-art deep learning models (CNN, EfficientNet, ResNet, ViT)
- 📊 **Severity Assessment** to quantify disease impact
- 🧠 **Explainability** via GradCAM heatmaps
- 💊 **Treatment Recommendations** based on disease and severity

---

## 🏗️ Architecture

```
Plant Guard AI
├── AI Module        → Model training, inference, explainability
├── Backend API      → FastAPI REST API for predictions
├── Frontend App     → React/TypeScript web interface
├── Datasets         → Curated dataset management
└── Docker           → Containerized deployment
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Node.js 20+
- Docker & Docker Compose

### Development Setup

```bash
# Clone the repository
git clone https://github.com/your-username/Plant_Guard_AI.git
cd Plant_Guard_AI

# Install Python dependencies
pip install -r requirements.txt

# Start backend
cd backend
uvicorn app.main:app --reload

# Start frontend (new terminal)
cd frontend
npm install
npm run dev

# Or start everything with Docker
cd docker
docker-compose up --build
```

---

## 📁 Project Structure

```
PLANT_GUARD_AI/
├── .github/workflows/    → CI/CD pipeline
├── ai/                   → Deep learning models & training
├── backend/              → FastAPI REST API
├── frontend/             → React TypeScript app
├── datasets/             → Raw and processed datasets
├── models/               → Saved model weights
├── notebooks/            → Jupyter notebooks
├── reports/              → Experiment reports
├── scripts/              → Automation scripts
├── tests/                → Test suites
├── docker/               → Docker configuration
└── docs/                 → Documentation
```

---

## 🧠 Supported Models

| Model | Architecture | Description |
|-------|-------------|-------------|
| CNN | Custom CNN | Lightweight baseline |
| EfficientNet | EfficientNet-B4 | Efficient & accurate |
| ResNet | ResNet-50 | Strong transfer learning |
| ViT | Vision Transformer | State-of-the-art accuracy |

---

## 📊 Datasets

- **PlantVillage** — 54,309 images, 38 disease classes
- **PlantDoc** — Real-world disease images with challenging conditions
- **Kaggle** — Additional curated plant disease datasets

---

## 🛣️ Roadmap

- [x] Phase 0: Project Structure Bootstrap
- [x] Phase 1: Data Pipeline & Preprocessing
- [ ] Phase 2: Model Training & Evaluation
- [ ] Phase 3: Backend API Development
- [ ] Phase 4: Frontend Development
- [ ] Phase 5: Deployment & CI/CD

---

## 🤝 Contributing

Contributions are welcome! Please read the contributing guidelines and open a pull request.

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.


📈 Development Progress
✅ Phase 0 — Project Bootstrap

Completed project architecture setup.

Created Structure
PLANT_GUARD_AI/

├── ai/
├── backend/
├── frontend/
├── datasets/
├── models/
├── reports/
├── scripts/
├── tests/
├── docs/
├── docker/
└── .github/
✅ Phase 1 — Dataset Collection & Analysis
Datasets Collected
Dataset	Images
PlantVillage	41,276
PlantVillage Extra	162,916
Total	204,192+
Script Executed
python scripts/analyze_dataset.py
Output
Total Datasets       : 3
Total Classes        : 3
Total Images         : 208,694
Corrupted Images     : 0
Generated Files
datasets/metadata/
├── dataset_summary.csv
└── corrupted_images.txt
✅ Phase 2.1 — Dataset Cleaning

Purpose:

Validate every image
Detect corrupted files
Remove invalid files
Quarantine problematic samples
Script Executed
python scripts/clean_dataset.py
Results
Total files scanned   : 208,699
Valid images          : 208,694
Corrupted images      : 0
Invalid files         : 5
Generated Files
datasets/metadata/
├── cleaning_report.csv
└── cleaning_summary.txt

datasets/quarantine/
└── 5 invalid files
✅ Phase 2.2 — Label Standardization

Purpose:

Convert all datasets into a unified class naming convention.

Example:

Mango healthy (P0a)
→
Mango___healthy

Mango diseased (P0b)
→
Mango___diseased

Tomato_Early_blight
→
Tomato___Early_blight
Script Executed
python scripts/standardize_labels.py
Results
Total discovered classes    : 211
Unique standardized classes : 64
Mappings generated          : 72
Generated Files
datasets/metadata/
└── class_mapping.json
✅ Phase 2.3 — Dataset Merge

Purpose:

Merge all datasets into a single standardized dataset.

Input
datasets/raw/
Output
datasets/intermediate/
Script Executed
python scripts/merge_datasets.py
Results
Classes Merged      : 210
Images Copied       : 208,686
Destination Classes : 64
Generated Files
datasets/metadata/
└── merge_report.csv
✅ Phase 2.4 — Dataset Splitting

Purpose:

Create train/validation/test sets.

Split Strategy
Train      80%
Validation 10%
Test       10%
Script Executed
python scripts/split_dataset.py
Input
datasets/intermediate/
Output
datasets/processed/

├── train/
├── val/
└── test/
Results
64 Classes
208,686 Images
Successfully split into:

Train Set
Validation Set
Test Set
Generated Files
datasets/metadata/
├── split_report.csv
└── split_summary.txt
📂 Current Dataset Pipeline
datasets/

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
    └── ...
🚧 Next Phase
Phase 3 — Dataset Statistics & Training Preparation

Upcoming tasks:

Calculate RGB Mean/Std
Compute Class Distribution
Detect Class Imbalance
Analyze Image Resolution Statistics
Generate Dataset Metadata

Planned Script:

python scripts/dataset_statistics.py

Outputs:

datasets/metadata/
├── dataset_statistics.json
└── class_distribution.csv
🎯 Current Status
Phase 0  Project Setup                    ✅
Phase 1  Dataset Collection               ✅
Phase 2.1 Dataset Cleaning                ✅
Phase 2.2 Label Standardization           ✅
Phase 2.3 Dataset Merge                   ✅
Phase 2.4 Train / Validation / Test Split ✅

Phase 3  Dataset Statistics               ⏳
Phase 4  DataLoader Pipeline              ⏳
Phase 5  Model Training                   ⏳
Phase 6  Evaluation                       ⏳
Phase 7  Explainability                   ⏳
Phase 8  Severity Assessment              ⏳
Phase 9  Backend API                      ⏳
Phase 10 Frontend & Deployment            ⏳