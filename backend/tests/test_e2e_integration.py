import io
import os
import sys
import uuid
from PIL import Image
import pytest

from app.core.config import settings
from app.db.models import AnalysisRecord
from app.ml.detector import ImageDetector
from app.ml.model_loader import ModelLoader
from app.schemas.analysis import AnalysisResponse, AnalysisListResponse
from tests.conftest import MockLoadedDetector


class DeterministicHeatmapDetector(ImageDetector):
    """
    Test-only detector providing deterministic forward-pass and
    Grad-CAM heatmap bytes through the ImageDetector abstraction.
    Does NOT use PyTorch, TensorFlow, or ONNX.
    """
    def __init__(self, ai_probability: float = 0.94):
        self.ai_probability = ai_probability

    def is_loaded(self) -> bool:
        return True

    def get_info(self) -> dict:
        return {
            "name": "Deterministic E2E Test Detector",
            "version": "e2e-test-v1",
            "task": "real-vs-ai-generated",
            "loaded": True,
            "device": "cpu"
        }

    def predict(self, image_tensor) -> dict:
        return {
            "raw_ai_probability": self.ai_probability,
            "raw_real_probability": 1.0 - self.ai_probability,
            "model_version": "e2e-test-v1"
        }

    def explain(self, image_tensor):
        # Create a deterministic 64x64 JPEG image to simulate a Grad-CAM heatmap
        img = Image.new("RGB", (64, 64), color=(255, 64, 64))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        return buf.getvalue()


def _get_stored_originals():
    originals_dir = settings.BASE_DIR / settings.originals_storage_path
    if not originals_dir.exists():
        return []
    return [f for f in originals_dir.glob("*") if f.is_file() and f.name != ".gitkeep"]


def _get_stored_heatmaps():
    heatmaps_dir = settings.BASE_DIR / settings.heatmaps_storage_path
    if not heatmaps_dir.exists():
        return []
    return [f for f in heatmaps_dir.glob("*") if f.is_file() and f.name != ".gitkeep"]


# =====================================================================
# 1. Full E2E Lifecycle: Likely AI-Generated Image
# =====================================================================
def test_e2e_likely_ai_generated_lifecycle(client, db_session, sample_jpeg_bytes):
    """
    Simulates a real frontend client upload:
    HTTP request -> multipart upload -> PIL decode -> metadata ->
    detector predict -> output validation -> calibration -> verdict ->
    DB persistence -> AnalysisResponse -> retrieval by ID -> history list.
    """
    original_detector = ModelLoader.get_detector()
    try:
        # Dependency injection of test-only mock detector
        mock_detector = MockLoadedDetector(ai_probability=0.88)
        ModelLoader.set_detector(mock_detector)

        # 1. Client uploads image via multipart POST
        response = client.post(
            "/api/v1/analyze",
            files={"image": ("../../traversal_photo.jpg", sample_jpeg_bytes, "image/jpeg")},
            data={"caption": "Synthetic portrait from social media"}
        )
        assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
        data = response.json()

        # 2. Verify all fields of AnalysisResponse
        parsed = AnalysisResponse(**data)
        assert parsed.analysis_id
        # Verify valid UUID format
        parsed_uuid = uuid.UUID(parsed.analysis_id)
        assert str(parsed_uuid) == parsed.analysis_id

        # Responsible verdict assertions
        assert parsed.verdict == "likely_ai_generated"
        assert parsed.verdict in ("likely_ai_generated", "likely_real")
        assert parsed.verdict not in ("fake", "definitely_fake", "100% AI")

        # Confidence bounds [0.0, 1.0]
        assert 0.0 <= parsed.confidence <= 1.0
        assert parsed.confidence == 0.88
        assert 0.0 <= parsed.threshold <= 1.0
        assert parsed.threshold == 0.50

        # Model and latency metadata
        assert parsed.model_version == "test-v1"
        assert isinstance(parsed.processing_time_ms, int)
        assert parsed.processing_time_ms >= 0

        # Calibration state
        assert parsed.is_calibrated is False
        assert parsed.calibration_method in ("uncalibrated", None)

        # Explanation details
        assert parsed.explanation.heatmap_available is False
        assert parsed.explanation.heatmap_url is None
        assert isinstance(parsed.explanation.cues, list)

        # Forensic metadata
        assert parsed.metadata.format == "JPEG"
        assert parsed.metadata.width == 128
        assert parsed.metadata.height == 128
        assert parsed.metadata.file_size == len(sample_jpeg_bytes)
        assert parsed.metadata.c2pa_detected is False
        assert parsed.metadata.c2pa_verified is False
        assert "supporting context only" in parsed.metadata.note

        # Ensure no internal path leakage
        assert "stored_path" not in data
        assert "c:\\" not in response.text.lower()
        assert "storage/originals" not in response.text.lower()

        # 3. Verify Database Persistence (GET /analyses/{analysis_id})
        get_res = client.get(f"/api/v1/analyses/{parsed.analysis_id}")
        assert get_res.status_code == 200
        get_data = get_res.json()
        assert get_data["analysis_id"] == parsed.analysis_id
        assert get_data["verdict"] == parsed.verdict
        assert get_data["confidence"] == parsed.confidence
        assert get_data["model_version"] == parsed.model_version

        # 4. Verify History List (GET /analyses)
        list_res = client.get("/api/v1/analyses?page=1&page_size=20")
        assert list_res.status_code == 200
        list_data = list_res.json()
        parsed_list = AnalysisListResponse(**list_data)
        assert parsed_list.total >= 1
        found_ids = [item.analysis_id for item in parsed_list.items]
        assert parsed.analysis_id in found_ids

        # 5. Verify Storage Isolation and Safety
        stored_files = _get_stored_originals()
        assert len(stored_files) == 1
        stored_file = stored_files[0]
        # Filename must be UUID-named only
        assert stored_file.name.endswith(".jpg")
        file_stem = stored_file.stem
        # Stem must be a valid UUID
        assert str(uuid.UUID(file_stem)) == file_stem
        # Malicious client filename must NOT dictate disk filename
        assert "traversal" not in stored_file.name
        # File must reside strictly inside originals directory
        originals_dir = (settings.BASE_DIR / settings.originals_storage_path).resolve()
        assert originals_dir in stored_file.resolve().parents

    finally:
        ModelLoader.set_detector(original_detector)


# =====================================================================
# 2. Full E2E Lifecycle: Likely Real Image (Responsible Vocabulary)
# =====================================================================
def test_e2e_likely_real_lifecycle(client, db_session, sample_png_bytes):
    """
    Simulates frontend client upload of an authentic image:
    Asserts responsible verdict vocabulary ('likely_real'), confidence
    calculation (1.0 - raw_ai_prob), and database record consistency.
    """
    original_detector = ModelLoader.get_detector()
    try:
        mock_detector = MockLoadedDetector(ai_probability=0.15)
        ModelLoader.set_detector(mock_detector)

        response = client.post(
            "/api/v1/analyze",
            files={"image": ("camera_snap.png", sample_png_bytes, "image/png")},
            data={"caption": "Outdoor nature photography"}
        )
        assert response.status_code == 200
        data = response.json()

        # Responsible verdict verification
        assert data["verdict"] == "likely_real"
        assert data["verdict"] not in ("authentic", "100% real", "certified")
        assert data["confidence"] == 0.85  # 1.0 - 0.15
        assert data["metadata"]["format"] == "PNG"

        # Verify DB record matches
        record = db_session.query(AnalysisRecord).filter(AnalysisRecord.id == data["analysis_id"]).first()
        assert record is not None
        assert record.verdict == "likely_real"
        assert record.confidence == 0.85
        assert record.filename == "camera_snap.png"

    finally:
        ModelLoader.set_detector(original_detector)


# =====================================================================
# 3. E2E Optional Heatmap Flow: Deterministic Visual Evidence
# =====================================================================
def test_e2e_heatmap_generation_and_retrieval(client, sample_jpeg_bytes):
    """
    Simulates end-to-end Grad-CAM visual evidence flow:
    Detector returns heatmap bytes -> saved to storage/heatmaps ->
    returned via GET /analyses/{id}/heatmap with Content-Type: image/jpeg.
    """
    original_detector = ModelLoader.get_detector()
    try:
        mock_detector = DeterministicHeatmapDetector(ai_probability=0.94)
        ModelLoader.set_detector(mock_detector)

        response = client.post(
            "/api/v1/analyze",
            files={"image": ("evidence_test.jpg", sample_jpeg_bytes, "image/jpeg")},
            data={"caption": "Forensic explainability analysis"}
        )
        assert response.status_code == 200
        data = response.json()
        analysis_id = data["analysis_id"]

        # Explainability schema checks
        assert data["explanation"]["heatmap_available"] is True
        assert data["explanation"]["heatmap_url"] == f"/api/v1/analyses/{analysis_id}/heatmap"

        # Verify heatmap stored on disk
        stored_heatmaps = _get_stored_heatmaps()
        assert len(stored_heatmaps) == 1
        heatmap_file = stored_heatmaps[0]
        assert heatmap_file.name == f"{analysis_id}.jpg"

        # Frontend requests heatmap image
        map_response = client.get(f"/api/v1/analyses/{analysis_id}/heatmap")
        assert map_response.status_code == 200
        assert "image/jpeg" in map_response.headers.get("content-type", "")
        # Verify valid JPEG binary header
        assert map_response.content[:3] == b"\xff\xd8\xff"
        assert len(map_response.content) > 100

    finally:
        ModelLoader.set_detector(original_detector)


# =====================================================================
# 4. E2E Missing Heatmap Handling
# =====================================================================
def test_e2e_missing_heatmap_returns_404(client, sample_jpeg_bytes):
    """
    When the active model architecture does not produce visual activation maps:
    Analysis succeeds (200), heatmap_available is False, and requesting
    the heatmap endpoint returns a safe 404 HEATMAP_NOT_FOUND.
    """
    original_detector = ModelLoader.get_detector()
    try:
        mock_detector = MockLoadedDetector(ai_probability=0.72)  # explain() returns None
        ModelLoader.set_detector(mock_detector)

        response = client.post(
            "/api/v1/analyze",
            files={"image": ("no_saliency.jpg", sample_jpeg_bytes, "image/jpeg")}
        )
        assert response.status_code == 200
        analysis_id = response.json()["analysis_id"]

        map_response = client.get(f"/api/v1/analyses/{analysis_id}/heatmap")
        assert map_response.status_code == 404
        assert map_response.json()["error"]["code"] == "HEATMAP_NOT_FOUND"

    finally:
        ModelLoader.set_detector(original_detector)


# =====================================================================
# 5. Model Abstraction Isolation (Zero ML Framework Coupling)
# =====================================================================
def test_e2e_model_abstraction_isolation():
    """
    Verifies architectural purity:
    The backend services and API routes MUST NOT import torch, torchvision,
    tensorflow, or onnx. They depend strictly on the ImageDetector interface.

    Scope note (.claude/specs/09-backend-ml-integration.md): the concrete
    detector *implementation* module(s) - e.g. app.ml.rgb_frequency_detector,
    which adapts the trained torch model behind the ImageDetector interface -
    are exclusively where a real ML framework dependency is expected to
    live, and are excluded from this check. Every other module under `app.`
    (services, API routes, core, db, schemas, and the ImageDetector interface
    itself in app.ml.detector/app.ml.model_loader/app.ml.calibration) must
    remain framework-free, matching this test's own stated intent.
    """
    forbidden_frameworks = ["torch", "torchvision", "tensorflow", "onnx", "onnxruntime"]
    ml_framework_adapter_modules = {"app.ml.rgb_frequency_detector"}

    # Check loaded modules under app
    for mod_name, mod in list(sys.modules.items()):
        if mod_name.startswith("app.") and mod is not None and mod_name not in ml_framework_adapter_modules:
            mod_file = getattr(mod, "__file__", "")
            if mod_file:
                with open(mod_file, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                    for fw in forbidden_frameworks:
                        # Allow mentions in comments/docstrings, but disallow active imports
                        lines = content.splitlines()
                        for line in lines:
                            stripped = line.strip()
                            if not stripped.startswith("#"):
                                assert not stripped.startswith(f"import {fw}"), \
                                    f"Forbidden import '{stripped}' found in {mod_file}"
                                assert not stripped.startswith(f"from {fw}"), \
                                    f"Forbidden import '{stripped}' found in {mod_file}"


# =====================================================================
# 6. Production State: Real Model Remains Safely Unavailable
# =====================================================================
def test_e2e_real_model_remains_unavailable(client, sample_jpeg_bytes, db_session):
    """
    Confirms default production configuration when no trained model weights exist:
    - GET /api/v1/model -> loaded=False
    - POST /api/v1/analyze -> 503 MODEL_UNAVAILABLE
    - Zero files created in storage/originals
    - Zero rows created in signalscope.db
    - Zero fake predictions emitted
    """
    # Reset singleton to default StubImageDetector
    ModelLoader._instance = None
    files_before = len(_get_stored_originals())
    records_before = db_session.query(AnalysisRecord).count()

    # 1. Model info check
    model_res = client.get("/api/v1/model")
    assert model_res.status_code == 200
    model_info = model_res.json()
    assert model_info["loaded"] is False
    assert model_info["name"] == "SignalScope Detector"

    # 2. Analyze attempt returns 503
    analyze_res = client.post(
        "/api/v1/analyze",
        files={"image": ("test.jpg", sample_jpeg_bytes, "image/jpeg")}
    )
    assert analyze_res.status_code == 503
    error_data = analyze_res.json()
    assert error_data["error"]["code"] == "MODEL_UNAVAILABLE"
    assert "not loaded" in error_data["error"]["message"].lower()

    # 3. Verify zero filesystem or database side effects
    assert len(_get_stored_originals()) == files_before
    assert db_session.query(AnalysisRecord).count() == records_before

