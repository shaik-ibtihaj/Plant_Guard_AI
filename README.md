# Plant Guard AI

Plant Guard AI is a research prototype for plant-leaf classification, visual
explanations, and mask-based severity measurement. It includes a PyTorch model,
a FastAPI service, a React + TypeScript interface, and sample images for trying
the upload workflow.

> **Research prototype only.** The checked-in EfficientNet-B0 checkpoint is a
> constrained benchmark, not a field-ready diagnostic. On its small held-out
> test subset it achieved 8.59% top-1 accuracy and 5.28% macro F1. Treat every
> prediction as an unreliable demonstration, not a diagnosis or treatment
> recommendation.

## Run locally on Windows

Use Python 3.11 and Node.js 20 or newer. Open Command Prompt or PowerShell in
the repository root (`Plant_Guard_AI`). Install Python dependencies into the
project virtual environment:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\python.exe -m ensurepip --upgrade
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt -r backend/requirements.txt
```

Install the frontend dependencies once:

```powershell
cd frontend
npm ci
```

Start the whole local app with one command from the `frontend` folder:

```powershell
npm run dev
```

This command starts the FastAPI backend using the repository's `.venv`, waits
for its health endpoint, then starts Vite. Open <http://localhost:5173>. The
backend API and interactive docs are at <http://127.0.0.1:8000> and
<http://127.0.0.1:8000/docs>. Keep the terminal open; press `Ctrl+C` to stop
both development servers.

If you already started the API yourself, `npm run dev` detects and reuses it.
To run the services manually instead, use two terminals:

```powershell
# Terminal 1: from the repository root
.venv\Scripts\python.exe -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```

```powershell
# Terminal 2: from the frontend folder
npm run dev:ui
```

### macOS/Linux

Create and install the Python environment from the repository root:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m ensurepip --upgrade
python -m pip install --upgrade pip
python -m pip install -r requirements.txt -r backend/requirements.txt
```

Then install and start the frontend:

```bash
cd frontend
npm ci
npm run dev
```

The same `npm run dev` command starts both API and frontend services.

## Try the sample photos

Ten generated PNG examples are in [`test_images/`](test_images/). Start the
app, choose **Diagnose plant**, and upload any file from that folder. The
images cover healthy tomato and pepper leaves and illustrative symptom
patterns for tomato early blight, apple scab/black rot, grape black rot, corn
common rust, potato late blight, peach bacterial spot, and strawberry leaf
scorch.

These generated pictures are workflow samples only: their filenames are not
verified ground-truth labels, and they must not be used to measure model
accuracy.

## What the app does

- **Diagnose:** upload a JPEG, PNG, or WEBP leaf image (up to 10 MB); view the
  top predictions and request a Grad-CAM/ViT visual explanation.
- **Severity:** upload an image plus same-size binary leaf and lesion masks;
  white mask pixels indicate foreground. Severity is based on measured area,
  not classifier confidence.
- **History:** browse or delete saved analysis metadata. Original image bytes
  are processed in memory and are not persisted.
- **Health:** `/health` checks process liveness; `/health/ready` checks model
  and database readiness.

The local database defaults to SQLite at `data/plant_guard.db`. Docker Compose
uses PostgreSQL. Local development allows API requests without a key. For
production, configure `PLANT_GUARD_API_KEY` with at least 32 characters and
set `APP_ENV=production`.

## Tests and build

From the repository root:

```powershell
.venv\Scripts\python.exe -m pytest -q tests
```

On macOS/Linux (with the venv activated):

```bash
python -m pytest -q tests
```

From `frontend`:

```powershell
npm test -- --run
npm run typecheck
npm run build
```

The Python suite covers data handling, metrics, evaluation, model setup, and
checkpoint utilities. Frontend tests cover image upload validation and display
formatting.

## Docker

Docker is optional for local development. If Docker Desktop and Compose are
installed:

```bash
cd docker
docker compose up --build
```

The stack exposes the frontend at <http://localhost:3000> and the API at
<http://localhost:8000>.

## Project layout

```text
ai/                 Training, models, evaluation, explainability, severity
backend/            FastAPI routes, inference, and history storage
frontend/           React UI, API proxy, and local development launcher
datasets/metadata/  Class mapping and image normalization statistics
models/             Selected checkpoint and model manifest
test_images/        Ten generated images for testing the upload workflow
tests/              Python tests
reports/            EDA, evaluation, and model comparison reports
docker/             Compose configuration
```

## Project status

Phases 1–10 have implementation artifacts in the repository: dataset
preparation/analysis, constrained model training and evaluation, Grad-CAM,
mask-based severity assessment, API, and frontend. Phase 11 deployment/CI
validation remains. The one-epoch checkpoint and incomplete field validation
are release blockers; implementation completion does not mean diagnostic
readiness.
