"""
Fetches the production model checkpoint before the backend starts, if it
is not already present locally.

Pure ops/deployment concern: does not modify any application code, model
architecture, or inference behavior. Reuses `app.core.config.settings` to
resolve the exact same weights path `ModelLoader` looks for (see
`app/ml/model_loader.py`), so there is one source of truth for where
weights belong.

No-ops (does nothing) if:
- MODEL_WEIGHTS_URL is not set (e.g. local development, where the
  checkpoint is already present on disk), or
- the destination file already exists (never re-downloads or overwrites
  an existing checkpoint).

Usage (before starting the server):
    python scripts/download_weights.py && uvicorn app.main:app --host 0.0.0.0 --port $PORT
"""

from __future__ import annotations

import os
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.config import settings  # noqa: E402


def resolve_weights_path() -> Path:
    weights_path = Path(settings.MODEL_WEIGHTS_PATH)
    if not weights_path.is_absolute():
        weights_path = settings.BASE_DIR / weights_path
    return weights_path


def download_weights() -> None:
    url = os.environ.get("MODEL_WEIGHTS_URL")
    destination = resolve_weights_path()

    if destination.exists():
        print(f"[download_weights] Weights already present at {destination}, skipping download.")
        return

    if not url:
        print(
            f"[download_weights] MODEL_WEIGHTS_URL not set and no weights file at {destination}. "
            "Skipping - the backend will start in stub-detector mode until weights are provided."
        )
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"[download_weights] Downloading model weights from {url} -> {destination} ...")
    tmp_path = destination.with_suffix(destination.suffix + ".partial")
    urllib.request.urlretrieve(url, tmp_path)
    tmp_path.rename(destination)
    size_mb = destination.stat().st_size / (1024 * 1024)
    print(f"[download_weights] Done: {destination} ({size_mb:.1f} MB).")


if __name__ == "__main__":
    download_weights()
