FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /app/backend/requirements.txt

COPY backend/app /app/backend/app
COPY ai /app/ai
COPY models/best_model.pth /app/models/best_model.pth
COPY models/production_model.json /app/models/production_model.json
COPY datasets/metadata/class_mapping.json /app/datasets/metadata/class_mapping.json
COPY datasets/metadata/rgb_stats.json /app/datasets/metadata/rgb_stats.json

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health/ready', timeout=3)"

CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
