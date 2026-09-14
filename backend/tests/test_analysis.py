import math
import uuid
import pytest
from app.core.config import settings
from app.db.models import AnalysisRecord
from app.ml.detector import ImageDetector
from app.ml.model_loader import ModelLoader
from tests.conftest import MockLoadedDetector


class ConfigurableTestDetector(ImageDetector):
    """Test detector allowing arbitrary return payloads or exceptions."""
    def __init__(self, output_payload=None, raise_exc=None):
        self.output_payload = output_payload
        self.raise_exc = raise_exc

    def is_loaded(self) -> bool:
        return True

    def get_info(self) -> dict:
        return {
            "name": "Configurable Test Detector",
            "version": "test-v1",
            "task": "real-vs-ai-generated",
            "loaded": True,
            "device": "cpu"
        }

    def predict(self, image_tensor) -> dict:
        if self.raise_exc:
            raise self.raise_exc
        return self.output_payload

    def explain(self, image_tensor):
        return None


def _get_stored_originals_count():
    """Helper to count stored original images excluding gitkeep."""
    originals_dir = settings.BASE_DIR / settings.originals_storage_path
    if not originals_dir.exists():
        return 0
    return len([f for f in originals_dir.glob("*") if f.is_file() and f.name != ".gitkeep"])


def test_analyze_without_model_weights_returns_503_and_no_side_effects(client, db_session, sample_jpeg_bytes):
    """
    (C, D, E) When model weights are absent:
    - POST /analyze -> 503 MODEL_UNAVAILABLE
    - No original image file is stored
    - No database record is created
    """
    ModelLoader._instance = None
    files_before = _get_stored_originals_count()
    records_before = db_session.query(AnalysisRecord).count()

    response = client.post(
        "/api/v1/analyze",
        files={"image": ("test.jpg", sample_jpeg_bytes, "image/jpeg")},
        data={"caption": "Test image"}
    )
    assert response.status_code == 503
    data = response.json()
    assert data["error"]["code"] == "MODEL_UNAVAILABLE"

    # Verify no filesystem artifacts created
    files_after = _get_stored_originals_count()
    assert files_after == files_before, "No original image should be stored when model is absent"

    # Verify no database records created
    records_after = db_session.query(AnalysisRecord).count()
    assert records_after == records_before, "No DB record should be created when model is absent"


def test_analyze_corrupt_file_returns_400(client, db_session):
    """Uploading unidentifiable binary returns 400 INVALID_IMAGE with no side effects."""
    files_before = _get_stored_originals_count()
    records_before = db_session.query(AnalysisRecord).count()

    response = client.post(
        "/api/v1/analyze",
        files={"image": ("bad.jpg", b"corrupt bytes data", "image/jpeg")}
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "INVALID_IMAGE"

    assert _get_stored_originals_count() == files_before
    assert db_session.query(AnalysisRecord).count() == records_before


def test_analyze_end_to_end_with_loaded_detector(client, db_session, sample_jpeg_bytes):
    """
    (F, G, H) End-to-end pipeline with loaded mock detector:
    - Complete analysis succeeds (HTTP 200)
    - Exactly one DB record created
    - Safe UUID-based file stored and exists on disk
    """
    original_detector = ModelLoader.get_detector()
    files_before = _get_stored_originals_count()
    records_before = db_session.query(AnalysisRecord).count()

    try:
        # 1. AI-Generated verdict test (prob 0.88 >= 0.50 threshold)
        ModelLoader.set_detector(MockLoadedDetector(ai_probability=0.88))
        res_ai = client.post(
            "/api/v1/analyze",
            files={"image": ("synthetic.jpg", sample_jpeg_bytes, "image/jpeg")},
            data={"caption": "Forensic test sample"}
        )
        assert res_ai.status_code == 200
        data_ai = res_ai.json()
        assert "analysis_id" in data_ai
        assert data_ai["verdict"] == "likely_ai_generated"
        assert data_ai["confidence"] == 0.88
        assert data_ai["threshold"] == 0.50
        assert data_ai["processing_time_ms"] >= 0
        assert data_ai["is_calibrated"] is False
        assert "metadata" in data_ai
        assert data_ai["metadata"]["format"] == "JPEG"
        assert data_ai["metadata"]["width"] == 128
        assert data_ai["metadata"]["height"] == 128
        assert "explanation" in data_ai
        assert data_ai["explanation"]["heatmap_available"] is False

        analysis_id = data_ai["analysis_id"]

        # Verify DB record (G)
        records_after = db_session.query(AnalysisRecord).count()
        assert records_after == records_before + 1
        record = db_session.query(AnalysisRecord).filter(AnalysisRecord.id == analysis_id).first()
        assert record is not None
        assert record.verdict == "likely_ai_generated"
        assert record.confidence == 0.88

        # Verify safe UUID file storage (H)
        files_after = _get_stored_originals_count()
        assert files_after == files_before + 1
        stored_path = settings.BASE_DIR / record.stored_path
        assert stored_path.exists()
        assert stored_path.is_file()
        assert analysis_id not in stored_path.name  # safe random UUID, not filename

        # 2. Retrieve analysis by ID
        res_get = client.get(f"/api/v1/analyses/{analysis_id}")
        assert res_get.status_code == 200
        data_get = res_get.json()
        assert data_get["analysis_id"] == analysis_id
        assert data_get["verdict"] == "likely_ai_generated"
        assert data_get["confidence"] == 0.88

        # 3. Real image verdict test (prob 0.15 < 0.50 threshold -> likely_real with 85% confidence)
        ModelLoader.set_detector(MockLoadedDetector(ai_probability=0.15))
        res_real = client.post(
            "/api/v1/analyze",
            files={"image": ("real.jpg", sample_jpeg_bytes, "image/jpeg")}
        )
        assert res_real.status_code == 200
        data_real = res_real.json()
        assert data_real["verdict"] == "likely_real"
        assert data_real["confidence"] == 0.85

    finally:
        ModelLoader.set_detector(original_detector)


@pytest.mark.parametrize(
    "invalid_prob,desc",
    [
        (float("nan"), "NaN probability"),
        (float("inf"), "+Infinity probability"),
        (float("-inf"), "-Infinity probability"),
        (-0.1, "Negative probability below 0.0"),
        (1.1, "Probability above 1.0"),
        ("0.85", "String probability"),
        (True, "Boolean value as probability"),
        (None, "None value as probability"),
    ],
)
def test_detector_output_validation_invalid_model_output(
    client, db_session, sample_jpeg_bytes, invalid_prob, desc
):
    """
    (I, J, K, L, M, N, O) Rejects invalid detector outputs with controlled HTTP 500 INVALID_MODEL_OUTPUT.
    Also verifies transaction rollback: newly stored original is cleaned up and DB is rolled back (Q, R).
    """
    original_detector = ModelLoader.get_detector()
    files_before = _get_stored_originals_count()
    records_before = db_session.query(AnalysisRecord).count()

    try:
        payload = {"raw_ai_probability": invalid_prob, "model_version": "test-v1"}
        ModelLoader.set_detector(ConfigurableTestDetector(output_payload=payload))

        response = client.post(
            "/api/v1/analyze",
            files={"image": ("test.jpg", sample_jpeg_bytes, "image/jpeg")}
        )
        assert response.status_code == 500
        data = response.json()
        assert data["error"]["code"] == "INVALID_MODEL_OUTPUT", f"Failed for {desc}"

        # Transaction safety: original image deleted, no record committed
        assert _get_stored_originals_count() == files_before, f"Image not cleaned up for {desc}"
        assert db_session.query(AnalysisRecord).count() == records_before, f"DB not rolled back for {desc}"

    finally:
        ModelLoader.set_detector(original_detector)


def test_detector_output_missing_field(client, db_session, sample_jpeg_bytes):
    """(N) Missing raw_ai_probability key returns controlled INVALID_MODEL_OUTPUT."""
    original_detector = ModelLoader.get_detector()
    files_before = _get_stored_originals_count()
    records_before = db_session.query(AnalysisRecord).count()

    try:
        # Output missing 'raw_ai_probability' completely
        payload = {"model_version": "test-v1"}
        ModelLoader.set_detector(ConfigurableTestDetector(output_payload=payload))

        response = client.post(
            "/api/v1/analyze",
            files={"image": ("test.jpg", sample_jpeg_bytes, "image/jpeg")}
        )
        assert response.status_code == 500
        data = response.json()
        assert data["error"]["code"] == "INVALID_MODEL_OUTPUT"

        assert _get_stored_originals_count() == files_before
        assert db_session.query(AnalysisRecord).count() == records_before

    finally:
        ModelLoader.set_detector(original_detector)


def test_detector_raises_unexpected_inference_exception(client, db_session, sample_jpeg_bytes):
    """
    (P) Unexpected inference exception returns controlled INFERENCE_ERROR (500)
    without leaking internal stack traces or paths.
    Also verifies storage cleanup and DB rollback (Q, R).
    """
    original_detector = ModelLoader.get_detector()
    files_before = _get_stored_originals_count()
    records_before = db_session.query(AnalysisRecord).count()

    try:
        ModelLoader.set_detector(
            ConfigurableTestDetector(raise_exc=RuntimeError("Simulated internal GPU/CUDA crash!"))
        )

        response = client.post(
            "/api/v1/analyze",
            files={"image": ("test.jpg", sample_jpeg_bytes, "image/jpeg")}
        )
        assert response.status_code == 500
        data = response.json()
        assert data["error"]["code"] == "INFERENCE_ERROR"
        # Verify stack trace or sensitive info is NOT leaked to client
        assert "Simulated internal GPU/CUDA crash!" not in data["error"]["message"]
        assert "Traceback" not in data["error"]["message"]

        # Storage & DB transactional cleanup
        assert _get_stored_originals_count() == files_before
        assert db_session.query(AnalysisRecord).count() == records_before

    finally:
        ModelLoader.set_detector(original_detector)


def test_get_analysis_not_found(client):
    """Requesting unknown analysis ID returns 404 ANALYSIS_NOT_FOUND."""
    random_id = str(uuid.uuid4())
    response = client.get(f"/api/v1/analyses/{random_id}")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "ANALYSIS_NOT_FOUND"


def test_list_analyses_pagination(client, sample_jpeg_bytes):
    """Verifies listing and pagination of analyses history."""
    original_detector = ModelLoader.get_detector()
    try:
        ModelLoader.set_detector(MockLoadedDetector(ai_probability=0.75))
        for i in range(3):
            client.post(
                "/api/v1/analyze",
                files={"image": (f"sample_{i}.jpg", sample_jpeg_bytes, "image/jpeg")}
            )

        res_list = client.get("/api/v1/analyses?page=1&page_size=2")
        assert res_list.status_code == 200
        data = res_list.json()
        assert data["total"] >= 3
        assert len(data["items"]) == 2
        assert data["page"] == 1
        assert data["page_size"] == 2
    finally:
        ModelLoader.set_detector(original_detector)


def test_get_heatmap_not_found(client, sample_jpeg_bytes):
    """When no heatmap was generated, endpoint returns 404."""
    original_detector = ModelLoader.get_detector()
    try:
        ModelLoader.set_detector(MockLoadedDetector(ai_probability=0.75))
        res = client.post(
            "/api/v1/analyze",
            files={"image": ("no_map.jpg", sample_jpeg_bytes, "image/jpeg")}
        )
        analysis_id = res.json()["analysis_id"]

        heatmap_res = client.get(f"/api/v1/analyses/{analysis_id}/heatmap")
        assert heatmap_res.status_code == 404
        assert heatmap_res.json()["error"]["code"] == "HEATMAP_NOT_FOUND"
    finally:
        ModelLoader.set_detector(original_detector)


def test_consistent_error_envelope_across_all_error_types(client, sample_jpeg_bytes):
    """
    (13) Verifies that all error responses follow the standardized error envelope:
    { "error": { "code": str, "message": str, "details": ... } }
    """
    # 1. 400 Invalid Image
    r1 = client.post("/api/v1/analyze", files={"image": ("bad.jpg", b"corrupted", "image/jpeg")})
    assert r1.status_code == 400
    d1 = r1.json()
    assert "error" in d1 and isinstance(d1["error"]["code"], str) and isinstance(d1["error"]["message"], str)

    # 2. 404 Nonexistent Analysis
    r2 = client.get(f"/api/v1/analyses/{uuid.uuid4()}")
    assert r2.status_code == 404
    d2 = r2.json()
    assert "error" in d2 and isinstance(d2["error"]["code"], str) and isinstance(d2["error"]["message"], str)

    # 3. 404 Unknown Route
    r3 = client.get("/api/v1/nonexistent_route")
    assert r3.status_code == 404
    d3 = r3.json()
    assert "error" in d3 and isinstance(d3["error"]["code"], str) and isinstance(d3["error"]["message"], str)

    # 4. 405 Method Not Allowed
    r4 = client.post("/api/v1/health")
    assert r4.status_code == 405
    d4 = r4.json()
    assert "error" in d4 and isinstance(d4["error"]["code"], str) and isinstance(d4["error"]["message"], str)

    # 5. 400/422 Validation Error (negative pagination page)
    r5 = client.get("/api/v1/analyses?page=0")
    assert r5.status_code in (400, 422)
    d5 = r5.json()
    assert "error" in d5 and isinstance(d5["error"]["code"], str) and isinstance(d5["error"]["message"], str)

    # 6. 503 Model Unavailable
    ModelLoader._instance = None
    r6 = client.post("/api/v1/analyze", files={"image": ("img.jpg", sample_jpeg_bytes, "image/jpeg")})
    assert r6.status_code == 503
    d6 = r6.json()
    assert "error" in d6 and isinstance(d6["error"]["code"], str) and isinstance(d6["error"]["message"], str)


def test_no_filesystem_path_leakage(client, sample_jpeg_bytes):
    """
    (14) Verifies that error and success responses never expose absolute filesystem paths.
    """
    base_dir_str = str(settings.BASE_DIR).lower()
    
    # Check 503 error
    ModelLoader._instance = None
    res_503 = client.post("/api/v1/analyze", files={"image": ("test.jpg", sample_jpeg_bytes, "image/jpeg")})
    assert base_dir_str not in res_503.text.lower()
    assert "c:\\" not in res_503.text.lower()

    # Check 404 error
    res_404 = client.get(f"/api/v1/analyses/{uuid.uuid4()}")
    assert base_dir_str not in res_404.text.lower()
    assert "c:\\" not in res_404.text.lower()

    # Check 200 success response
    original_detector = ModelLoader.get_detector()
    try:
        ModelLoader.set_detector(MockLoadedDetector(0.8))
        res_200 = client.post("/api/v1/analyze", files={"image": ("test.jpg", sample_jpeg_bytes, "image/jpeg")})
        assert res_200.status_code == 200
        assert base_dir_str not in res_200.text.lower()
        assert "c:\\" not in res_200.text.lower()
        # Ensure stored_path is not exposed in public AnalysisResponse
        assert "stored_path" not in res_200.json()
    finally:
        ModelLoader.set_detector(original_detector)


def test_no_stack_trace_leakage(client, sample_jpeg_bytes):
    """
    (15) Verifies that internal exception details and tracebacks are never exposed to the client.
    """
    original_detector = ModelLoader.get_detector()
    try:
        ModelLoader.set_detector(
            ConfigurableTestDetector(raise_exc=ZeroDivisionError("division by zero in tensor core"))
        )
        response = client.post(
            "/api/v1/analyze",
            files={"image": ("test.jpg", sample_jpeg_bytes, "image/jpeg")}
        )
        assert response.status_code == 500
        body = response.text
        assert "Traceback" not in body
        assert "ZeroDivisionError" not in body
        assert "division by zero in tensor core" not in body
        assert response.json()["error"]["code"] == "INFERENCE_ERROR"
    finally:
        ModelLoader.set_detector(original_detector)

