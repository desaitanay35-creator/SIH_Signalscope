# 🔍 SignalScope: AI-Generated Image Detection & Explainability Tool
# SignalScope

SignalScope is a modular, high-performance forensic toolkit designed to detect AI-generated imagery (GANs, Diffusion models, Midjourney, DALL-E 3) and provide deep visual explainability (Grad-CAM, FFT spectrums, Error Level Analysis).
> **Telling Real From Synthetic in the Age of Generative Media.**

SignalScope is an AI and media-forensics application designed to assess whether an image is:
1. **Likely real**
2. **Likely AI-generated**

The system provides calibrated likelihood assessments, model version information, processing latency, forensic metadata (EXIF / C2PA markers), and visual explainability evidence (Grad-CAM heatmaps).

---

## 🏛 Repository Architecture & Team Allocation
## Repository Structure

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
```
SIH_Signalscope/
├── backend/                        # FastAPI Backend Application
│   ├── app/
│   │   ├── main.py                 # FastAPI app entrypoint, lifespan, CORS, error handling
│   │   ├── api/routes/             # /health, /model, /analyze, /analyses
│   │   ├── core/                   # Config (pydantic-settings), logging, custom error types
│   │   ├── schemas/                # Pydantic request/response schemas
│   │   ├── services/               # Image validation, inference, metadata, storage, analysis
│   │   ├── ml/                     # ImageDetector interface, Stub detector, Calibrator
│   │   └── db/                     # SQLite / SQLAlchemy models & session
│   ├── model/weights/              # Directory for ML model weights
│   ├── storage/                    # Safe UUID storage for originals and heatmaps
│   ├── tests/                      # Pytest automated test suite (15 passing tests)
│   ├── requirements.txt            # Backend dependencies
│   ├── .env.example                # Environment variable configuration template
│   ├── .gitignore                  # Storage, db, and weights ignore rules
│   └── README.md                   # Backend documentation
├── .gitignore                      # Git ignore rules
└── README.md                       # Root documentation
```

---

## 🚀 Quick Start Guide
## Quickstart: Running the Backend

### 1. CLI Prediction (`model/predict.py`)
### 1. Install Dependencies
```bash
python model/predict.py --image path/to/sample.jpg
cd backend
pip install -r requirements.txt
```

### 2. Run API Service
### 2. Configure Environment
```bash
uvicorn api.main:app --reload --port 8000
copy .env.example .env
```

### 3. Run Streamlit UI
### 3. Start the FastAPI Server
```bash
streamlit run ui/app.py
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

### 4. Open Swagger Documentation
With the server running, navigate in your browser to:
- **Swagger UI**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **ReDoc**: [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)

---

## 🛠 Setup Repository Script
To recreate the repository folder structure automatically, run:
## API Endpoints (`/api/v1`)

- `GET /api/v1/health`: System health and model loaded status (`{"status": "ok", "model_loaded": false}`).
- `GET /api/v1/model`: Active detector metadata (`name`, `version`, `task`, `loaded`).
- `POST /api/v1/analyze`: Upload image (`multipart/form-data`) for forensic analysis.
- `GET /api/v1/analyses/{analysis_id}`: Retrieve a stored analysis record by UUID.
- `GET /api/v1/analyses`: Retrieve paginated history of past analyses.
- `GET /api/v1/analyses/{analysis_id}/heatmap`: Download Grad-CAM visual heatmap if generated.

---

## ML Model Integration

The backend is decoupled from model architecture specifics. Any ML model (ResNet, EfficientNet, ViT, or custom backbones) can be integrated by implementing the `ImageDetector` contract (`backend/app/ml/detector.py`).

When model weights are not loaded in `backend/model/weights/`:
- `GET /api/v1/health` reports `model_loaded: false`
- `POST /api/v1/analyze` returns HTTP `503 Service Unavailable` (`MODEL_UNAVAILABLE`)
- **No fake results, placeholder predictions, or hardcoded probabilities are returned.**

---

## Running Tests

From the `backend` directory:
```bash
python setup_repo.py
cd backend
python -m pytest tests/ -v
```

All 15 automated unit and integration tests run without external dependencies and pass in `< 1` second.

---

## Responsible-Use Policy

- Outputs are strictly **likelihood assessments** (`"Likely AI-generated"` / `"Likely real"`).
- Never makes definitive or unprovable claims (*"100% fake"* / *"definitely AI"*).
- Not designed to profile real individuals or adjudicate political claims.
- Metadata (EXIF/C2PA) serves as contextual supporting evidence only.
