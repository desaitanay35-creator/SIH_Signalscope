# SignalScope Backend

> **Telling Real From Synthetic in the Age of Generative Media.**

SignalScope is an AI and media-forensics inference application designed to assess whether an image is **Likely real** or **Likely AI-generated**, providing likelihood probability scores, forensic metadata, and grounded visual explainability evidence.

> [!IMPORTANT]
> **ML Integration Status**: The production detector is pending ML model integration from the independent ML branch. Model weights will be placed under `backend/model/weights/`. Currently, `/api/v1/analyze` safely returns HTTP `503 MODEL_UNAVAILABLE` until real weights are supplied. **No fake production predictions or synthetic heatmaps are emitted.**

---

## 1. Quickstart for Frontend Developers

### Starting the Server
From the `backend` directory:
```powershell
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

### Interactive API Documentation
- **Swagger UI**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **ReDoc**: [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)
- **OpenAPI Spec**: [http://127.0.0.1:8000/openapi.json](http://127.0.0.1:8000/openapi.json)

### CORS Support
CORS is configured out-of-the-box for local frontend development (`*` by default). You can restrict allowed origins via the `CORS_ORIGINS` environment variable in `backend/.env`:
```env
CORS_ORIGINS=["http://localhost:3000", "http://localhost:5173"]
```

---

## 2. API Contract & Integration Reference

All endpoints are versioned under `/api/v1`.

### 1. Health Check
`GET /api/v1/health`

Confirms the API is online and indicates whether the ML model is operational.
```json
{
  "status": "ok",
  "model_loaded": false
}
```

---

### 2. Model Information
`GET /api/v1/model`

Returns metadata about the active ML detector module without leaking server paths.
```json
{
  "name": "SignalScope Detector",
  "version": "signalscope-v1",
  "task": "real-vs-ai-generated",
  "loaded": false,
  "device": "unassigned"
}
```

---

### 3. Analyze Image
`POST /api/v1/analyze`

Submits an image for media forensics analysis.

- **Content-Type**: `multipart/form-data`
- **Parameters**:
  - `image` (*required*, binary file): Supported formats: `JPEG`, `PNG`, `WEBP`. Maximum size: `10MB`. Dimensions: $32 \times 32\text{px}$ to $8192 \times 8192\text{px}$.
  - `caption` (*optional*, string): Accompanying text context or user caption.

#### Example `curl` Request:
```bash
curl -X POST http://127.0.0.1:8000/api/v1/analyze \
  -F "image=@sample.jpg;type=image/jpeg" \
  -F "caption=Photo from social media feed"
```

#### Successful Response (`200 OK` — `AnalysisResponse`):
```json
{
  "analysis_id": "8f3b2024-5dc3-455b-bf42-2d1844b2f270",
  "verdict": "likely_ai_generated",
  "confidence": 0.88,
  "threshold": 0.50,
  "model_version": "signalscope-v1",
  "processing_time_ms": 142,
  "is_calibrated": false,
  "calibration_method": "uncalibrated",
  "explanation": {
    "summary": "Grounded visual heatmap generated from model activation layers.",
    "cues": [],
    "heatmap_url": "/api/v1/analyses/8f3b2024-5dc3-455b-bf42-2d1844b2f270/heatmap",
    "heatmap_available": true
  },
  "metadata": {
    "format": "JPEG",
    "width": 1024,
    "height": 768,
    "file_size": 245890,
    "has_exif": true,
    "exif_summary": {
      "camera_make": "Canon",
      "camera_model": "EOS 80D"
    },
    "c2pa_detected": false,
    "c2pa_verified": false,
    "has_c2pa": false,
    "note": "Metadata provides supporting context only and does not determine authenticity."
  },
  "created_at": "2026-09-14T07:30:00.000000Z"
}
```

---

### 4. Analysis History (Paginated)
`GET /api/v1/analyses?page=1&page_size=20`

Retrieves a paginated list of past analyses ordered chronologically descending.

- **Query Parameters**:
  - `page`: Integer $\ge 1$ (default: `1`).
  - `page_size`: Integer between `1` and `100` (default: `20`).

#### Response (`200 OK` — `AnalysisListResponse`):
```json
{
  "items": [
    {
      "analysis_id": "8f3b2024-5dc3-455b-bf42-2d1844b2f270",
      "filename": "sample.jpg",
      "verdict": "likely_ai_generated",
      "confidence": 0.88,
      "model_version": "signalscope-v1",
      "processing_time_ms": 142,
      "created_at": "2026-09-14T07:30:00.000000Z"
    }
  ],
  "total": 1,
  "page": 1,
  "page_size": 20
}
```

---

### 5. Single Analysis Result
`GET /api/v1/analyses/{analysis_id}`

Retrieves the complete `AnalysisResponse` record by UUID.
- If analysis exists: `200 OK` (returns full `AnalysisResponse`).
- If UUID is nonexistent: `404 Not Found` (`ANALYSIS_NOT_FOUND`).
- If UUID format is malformed: `400 Bad Request` (`INVALID_ANALYSIS_ID`).

---

### 6. Grad-CAM Heatmap Image
`GET /api/v1/analyses/{analysis_id}/heatmap`

Streams the visual Grad-CAM heatmap JPEG file for frontend visualization overlay.
- If heatmap was generated: `200 OK` (`Content-Type: image/jpeg`).
- If analysis does not exist: `404 Not Found` (`ANALYSIS_NOT_FOUND`).
- If analysis exists but no heatmap generated: `404 Not Found` (`HEATMAP_NOT_FOUND`).
- If UUID format is malformed: `400 Bad Request` (`INVALID_ANALYSIS_ID`).

---

## 3. Standardized Error Handling

All error responses across all endpoints follow a uniform envelope:

```json
{
  "error": {
    "code": "ERROR_CODE",
    "message": "Human-readable description of the error.",
    "details": null
  }
}
```

### Canonical Error Codes:

| HTTP Status | Error Code | Description |
|---|---|---|
| `400` | `INVALID_IMAGE` | Payload cannot be decoded as an image, has empty bytes, or has dimensions $<32\text{px}$ or $>8192\text{px}$. |
| `400` | `UNSUPPORTED_IMAGE_TYPE` | Format is not in allowed list (`JPEG`, `PNG`, `WEBP`). |
| `400` | `INVALID_ANALYSIS_ID` | The supplied analysis ID is not a valid UUID format. |
| `400` | `VALIDATION_ERROR` | Request payload or query parameters failed validation (e.g. `page < 1` or `page_size > 100`). |
| `404` | `ANALYSIS_NOT_FOUND` | No record matching the given UUID exists in the database. |
| `404` | `HEATMAP_NOT_FOUND` | Analysis exists but no visual heatmap was generated or stored. |
| `404` | `NOT_FOUND` | Endpoint or route does not exist. |
| `405` | `METHOD_NOT_ALLOWED` | HTTP method not permitted on the route. |
| `413` | `FILE_TOO_LARGE` | Uploaded file size exceeds the configured maximum limit of `10MB`. |
| `500` | `INVALID_MODEL_OUTPUT` | Detector produced an invalid probability (e.g., non-numeric, infinite, or out of range $[0.0, 1.0]$). |
| `500` | `INFERENCE_ERROR` | An unexpected failure occurred during model preprocessing or inference. |
| `500` | `DATABASE_ERROR` | A database query or transaction failed. |
| `500` | `INTERNAL_SERVER_ERROR` | General unhandled server error. |
| `503` | `MODEL_UNAVAILABLE` | Detector weights are not loaded. |

---

## 4. Architecture & Security Guarantees

```
backend/
├── app/
│   ├── main.py                     # FastAPI application factory, lifespan, CORS, error handlers
│   ├── api/
│   │   └── routes/
│   │       ├── health.py           # GET /api/v1/health
│   │       ├── model.py            # GET /api/v1/model
│   │       └── analysis.py         # /analyze, /analyses, /analyses/{id}, /analyses/{id}/heatmap
│   ├── core/
│   │   ├── config.py               # Settings (Pydantic BaseSettings, configurable normalization)
│   │   ├── logging.py              # Structured logging (zero sensitive data leakage)
│   │   └── errors.py               # AppException hierarchy
│   ├── schemas/
│   │   ├── common.py               # Standardized ErrorResponse schema
│   │   └── analysis.py             # DTO schemas (AnalysisResponse, AnalysisListResponse, etc.)
│   ├── services/
│   │   ├── image_service.py        # PIL decoding, RGB standardization, decompression bomb check
│   │   ├── inference_service.py    # Tensor resizing and normalization
│   │   ├── metadata_service.py     # EXIF extraction and C2PA provenance detection
│   │   ├── explanation_service.py  # Grad-CAM heatmap and cue management
│   │   ├── storage_service.py      # UUID-based storage with strict path-containment validation
│   │   └── analysis_service.py     # 15-step transaction pipeline orchestration
│   ├── ml/
│   │   ├── detector.py             # ImageDetector ABC contract and StubImageDetector
│   │   ├── model_loader.py         # Singleton weights manager
│   │   └── calibration.py          # Calibrator & responsible verdict logic
│   └── db/
│       ├── database.py             # SQLAlchemy session manager (SQLite default)
│       └── models.py               # AnalysisRecord ORM table
├── model/
│   └── weights/                    # Storage directory for trained weights (*.pth, *.onnx)
├── storage/
│   ├── originals/                  # Safe UUID-stored originals
│   └── heatmaps/                   # Generated Grad-CAM heatmaps
└── tests/                          # 66 comprehensive automated tests
```

- **Zero Path Traversal**: Client filenames are never used for filesystem operations. Uploaded files are saved as `<uuid>.<ext>`.
- **Zero Information Leakage**: Responses never leak server disk paths (`C:\`), Python tracebacks, or SQL queries.
- **Decompression Bomb Protection**: Pillow's `MAX_IMAGE_PIXELS` ($67.1\text{M}$ pixels) blocks malicious pathological image payloads.
- **Strict ML Boundary**: The backend services and routes depend exclusively on the `ImageDetector` abstract base class. When trained model weights are ready, the ML teammate only needs to hook their detector class into `ModelLoader.load_detector()`.

---

## 5. Running the Tests

To run the full automated test suite:
```powershell
cd backend
python -m pytest
```
Testing covers health endpoints, model metadata, corrupt/malformed image handling, decompression bombs, safe UUID path traversal rejection, pagination bounds, transactional cleanups, and the frontend API contract.
