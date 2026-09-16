# SignalScope backend — Cloud Run deployment image.
#
# Deployment-only artifact: does not modify backend/, model/, config/, or
# frontend/ code. Mirrors the existing Railway build/start flow
# (.railway/railway.ts) so behavior is identical across both platforms.

FROM python:3.12-slim

WORKDIR /app

# Install root requirements (torch/torchvision CPU wheels) first, then
# backend requirements. Same two-step install Railway performs, in the
# same order, so resolved versions match the existing deployment.
COPY requirements.txt ./requirements.txt
COPY backend/requirements.txt ./backend/requirements.txt

RUN pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r backend/requirements.txt

# Preserve the repository-root layout: backend/app/ml imports model.* and
# config.* by bootstrapping the repo root onto sys.path at runtime
# (see backend/app/ml/rgb_frequency_detector.py), so those directories
# must exist at the same relative locations as in source control.
COPY backend/ ./backend/
COPY model/ ./model/
COPY config/ ./config/

# No model weights, no SQLite DB, no uploaded/heatmap files are copied in
# (excluded via .dockerignore) — weights are fetched at container startup.

EXPOSE 8080

WORKDIR /app/backend

CMD ["sh", "-c", "python scripts/download_weights.py && uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8080}"]
