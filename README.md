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
- [ ] Phase 1: Data Pipeline & Preprocessing
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
