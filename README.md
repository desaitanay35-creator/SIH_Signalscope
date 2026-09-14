# 🔍 SignalScope: AI-Generated Image Detection & Explainability Tool

SignalScope is a modular, high-performance forensic toolkit designed to detect AI-generated imagery (GANs, Diffusion models, Midjourney, DALL-E 3) and provide deep visual explainability (Grad-CAM, FFT spectrums, Error Level Analysis).

---

## 🏛 Repository Architecture & Team Allocation

| Module | File Path | Team Member | Primary Responsibility |
| :--- | :--- | :--- | :--- |
| **Model** | `model/backbone.py` | **Member 1** | Deep neural network classifier architecture |
| **Model** | `model/loader.py` | **Member 1** | Model weights loading & device allocation |
| **Model** | `model/feature_extractor.py` | **Member 1** | Latent feature vector & embedding extraction |
| **Model** | `model/predict.py` | **Member 1** | **Mandatory CLI Prediction Interface** |
| **XAI** | `explainability/gradcam.py` | **Member 2** | Grad-CAM activation heatmap visualizer |
| **XAI** | `explainability/fft_analysis.py` | **Member 2** | 2D FFT spectral artifact analyzer |
| **XAI** | `explainability/ela_analysis.py` | **Member 2** | Error Level Analysis (JPEG re-compression) |
| **XAI** | `explainability/attention_map.py` | **Member 2** | ViT spatial self-attention map extractor |
| **XAI** | `explainability/report_generator.py` | **Member 2** | Consolidates XAI signals into unified report |
| **Data** | `data/preprocessor.py` | **Member 3** | Image resizing, normalization & tensor conversion |
| **Data** | `data/face_detector.py` | **Member 3** | Face detection & facial crop alignment |
| **Data** | `data/noise_extractor.py` | **Member 3** | High-pass residual & noise pattern extraction |
| **Data** | `data/dataset_loader.py` | **Member 3** | PyTorch Dataset wrapper for benchmark testing |
| **Data** | `data/augmentations.py` | **Member 3** | Social media compression & blur degradations |
| **API** | `api/main.py` | **Member 4** | FastAPI entrypoint & router initialization |
| **API** | `api/schemas.py` | **Member 4** | Pydantic request/response model validation |
| **API** | `api/endpoints.py` | **Member 4** | REST endpoints for single image & batch upload |
| **API** | `api/task_queue.py` | **Member 4** | Async background task queue manager |
| **API** | `api/middleware.py` | **Member 4** | CORS, security & performance timing middleware |
| **UI** | `ui/app.py` | **Member 5** | Streamlit web application dashboard entrypoint |
| **UI** | `ui/components.py` | **Member 5** | Reusable UI widgets & metric badges |
| **UI** | `ui/heatmap_renderer.py` | **Member 5** | Image & Grad-CAM visual blending renderer |
| **UI** | `ui/report_viewer.py` | **Member 5** | Detailed forensic evidence dashboard view |
| **Ops** | `config/settings.py` | **Member 6** | System settings & environment configuration |
| **Ops** | `utils/logger.py` | **Member 6** | Structured logging handler |
| **Ops** | `utils/io.py` | **Member 6** | Image file loading & format validation |
| **Ops** | `utils/metrics.py` | **Member 6** | ROC-AUC, Accuracy & ECE metric calculations |
| **Ops** | `tests/*` | **Member 6** | PyTest test suite (unit & integration tests) |
| **Ops** | `Dockerfile` | **Member 6** | Container definition & Docker Compose configuration |

---

## 🚀 Quick Start Guide

### 1. CLI Prediction (`model/predict.py`)
```bash
python model/predict.py --image path/to/sample.jpg
```

### 2. Run API Service
```bash
uvicorn api.main:app --reload --port 8000
```

### 3. Run Streamlit UI
```bash
streamlit run ui/app.py
```

---

## 🛠 Setup Repository Script
To recreate the repository folder structure automatically, run:
```bash
python setup_repo.py
```
