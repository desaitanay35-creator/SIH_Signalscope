import io
import uuid
from PIL import Image
import pytest

from app.core.config import settings
from app.ml.detector import ImageDetector
from app.ml.model_loader import ModelLoader
from app.schemas.analysis import AnalysisResponse, AnalysisListResponse
from tests.conftest import MockLoadedDetector


class MockDetectorWithHeatmap(ImageDetector):
    """Mock detector that produces grounded heatmap bytes for explainability testing."""
    def __init__(self, ai_probability: float = 0.92):
        self.ai_probability = ai_probability

    def is_loaded(self) -> bool:
        return True

    def get_info(self) -> dict:
        return {
            "name": "Mock Heatmap Detector",
            "version": "test-cam-v1",
            "task": "real-vs-ai-generated",
            "loaded": True,
            "device": "cpu"
        }

    def predict(self, image_tensor) -> dict:
        return {
            "raw_ai_probability": self.ai_probability,
            "raw_real_probability": 1.0 - self.ai_probability,
            "model_version": "test-cam-v1"
        }

    def explain(self, image_tensor):
        # Generate valid JPEG bytes representing Grad-CAM activation heatmap
        img = Image.new("RGB", (64, 64), color=(255, 0, 0))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        return buf.getvalue()


# =====================================================================
# 1. Analyze success with MockLoadedDetector
# =====================================================================
def test_frontend_analyze_success(client, sample_jpeg_bytes):
    original_detector = ModelLoader.get_detector()
    try:
        ModelLoader.set_detector(MockLoadedDetector(ai_probability=0.88))
        response = client.post(
            "/api/v1/analyze",
            files={"image": ("test_upload.jpg", sample_jpeg_bytes, "image/jpeg")},
            data={"caption": "Test synthetic image"}
        )
        assert response.status_code == 200
        data = response.json()
        assert "analysis_id" in data
        assert data["verdict"] == "likely_ai_generated"
        assert data["confidence"] == 0.88
    finally:
        ModelLoader.set_detector(original_detector)


# =====================================================================
# 2. Response strictly conforms to AnalysisResponse schema
# =====================================================================
def test_frontend_response_conforms_to_schema(client, sample_jpeg_bytes):
    original_detector = ModelLoader.get_detector()
    try:
        ModelLoader.set_detector(MockLoadedDetector(ai_probability=0.25))
        response = client.post(
            "/api/v1/analyze",
            files={"image": ("real_sample.jpg", sample_jpeg_bytes, "image/jpeg")},
            data={"caption": "Authentic photo"}
        )
        assert response.status_code == 200
        data = response.json()

        # Validate against Pydantic schema
        parsed = AnalysisResponse(**data)
        assert parsed.verdict == "likely_real"
        assert parsed.confidence == 0.75  # 1.0 - 0.25
        assert parsed.threshold == 0.50
        assert parsed.model_version == "test-v1"
        assert parsed.processing_time_ms >= 0
        assert isinstance(parsed.is_calibrated, bool)
        assert parsed.calibration_method in ("uncalibrated", None)

        # Explanation contract
        assert isinstance(parsed.explanation.summary, (str, type(None)))
        assert isinstance(parsed.explanation.cues, list)
        assert isinstance(parsed.explanation.heatmap_available, bool)

        # Metadata contract
        assert parsed.metadata.format == "JPEG"
        assert parsed.metadata.width > 0
        assert parsed.metadata.height > 0
        assert parsed.metadata.file_size > 0
        assert isinstance(parsed.metadata.has_exif, bool)
        assert isinstance(parsed.metadata.c2pa_detected, bool)
        assert isinstance(parsed.metadata.c2pa_verified, bool)
        assert "Metadata provides supporting context only" in parsed.metadata.note

        # Confirm no internal model or path leakage
        assert "stored_path" not in data
        assert "raw_bytes" not in data
    finally:
        ModelLoader.set_detector(original_detector)


# =====================================================================
# 3. History response conforms to AnalysisListResponse
# =====================================================================
def test_frontend_history_conforms_to_schema(client, sample_jpeg_bytes):
    original_detector = ModelLoader.get_detector()
    try:
        ModelLoader.set_detector(MockLoadedDetector(ai_probability=0.90))
        client.post(
            "/api/v1/analyze",
            files={"image": ("history_sample.jpg", sample_jpeg_bytes, "image/jpeg")}
        )

        response = client.get("/api/v1/analyses?page=1&page_size=10")
        assert response.status_code == 200
        data = response.json()

        parsed = AnalysisListResponse(**data)
        assert parsed.total >= 1
        assert parsed.page == 1
        assert parsed.page_size == 10
        assert len(parsed.items) >= 1

        item = parsed.items[0]
        assert item.analysis_id
        assert item.filename == "history_sample.jpg"
        assert item.verdict in ("likely_ai_generated", "likely_real")
        assert 0.0 <= item.confidence <= 1.0
        assert item.model_version
        assert item.processing_time_ms >= 0
        assert item.created_at
    finally:
        ModelLoader.set_detector(original_detector)


# =====================================================================
# 4. Pagination works correctly (multi-page, offsets, page sizes)
# =====================================================================
def test_frontend_pagination_behavior(client, sample_jpeg_bytes):
    original_detector = ModelLoader.get_detector()
    try:
        ModelLoader.set_detector(MockLoadedDetector(ai_probability=0.75))
        for i in range(5):
            client.post(
                "/api/v1/analyze",
                files={"image": (f"page_test_{i}.jpg", sample_jpeg_bytes, "image/jpeg")}
            )

        # Page 1: 2 items
        r1 = client.get("/api/v1/analyses?page=1&page_size=2")
        assert r1.status_code == 200
        d1 = r1.json()
        assert d1["page"] == 1
        assert d1["page_size"] == 2
        assert len(d1["items"]) == 2
        first_page_ids = [it["analysis_id"] for it in d1["items"]]

        # Page 2: 2 items (different IDs)
        r2 = client.get("/api/v1/analyses?page=2&page_size=2")
        assert r2.status_code == 200
        d2 = r2.json()
        assert d2["page"] == 2
        assert d2["page_size"] == 2
        assert len(d2["items"]) == 2
        second_page_ids = [it["analysis_id"] for it in d2["items"]]

        # Verify page isolation (no overlapping IDs between consecutive pages)
        assert set(first_page_ids).isdisjoint(set(second_page_ids))
    finally:
        ModelLoader.set_detector(original_detector)


# =====================================================================
# 5. Single analysis response retrieves full record
# =====================================================================
def test_frontend_single_analysis_response(client, sample_jpeg_bytes):
    original_detector = ModelLoader.get_detector()
    try:
        ModelLoader.set_detector(MockLoadedDetector(ai_probability=0.88))
        res_post = client.post(
            "/api/v1/analyze",
            files={"image": ("single_fetch.jpg", sample_jpeg_bytes, "image/jpeg")},
            data={"caption": "Detailed examination"}
        )
        assert res_post.status_code == 200
        analysis_id = res_post.json()["analysis_id"]

        res_get = client.get(f"/api/v1/analyses/{analysis_id}")
        assert res_get.status_code == 200
        data = res_get.json()

        # Validate identical fields
        assert data["analysis_id"] == analysis_id
        assert data["verdict"] == "likely_ai_generated"
        assert data["confidence"] == 0.88
        assert data["metadata"]["format"] == "JPEG"
    finally:
        ModelLoader.set_detector(original_detector)


# =====================================================================
# 6. Nonexistent analysis returns 404 ANALYSIS_NOT_FOUND
# =====================================================================
def test_frontend_nonexistent_analysis_returns_404(client):
    nonexistent = str(uuid.uuid4())
    res = client.get(f"/api/v1/analyses/{nonexistent}")
    assert res.status_code == 404
    data = res.json()
    assert data["error"]["code"] == "ANALYSIS_NOT_FOUND"
    assert "not found" in data["error"]["message"].lower()


# =====================================================================
# 7. Invalid analysis ID returns 400 INVALID_ANALYSIS_ID
# =====================================================================
def test_frontend_invalid_analysis_id_returns_400(client):
    res = client.get("/api/v1/analyses/bad-uuid-format")
    assert res.status_code == 400
    data = res.json()
    assert data["error"]["code"] == "INVALID_ANALYSIS_ID"
    assert "UUID format" in data["error"]["message"]


# =====================================================================
# 8. Missing heatmap returns 404 HEATMAP_NOT_FOUND
# =====================================================================
def test_frontend_missing_heatmap_returns_404(client, sample_jpeg_bytes):
    original_detector = ModelLoader.get_detector()
    try:
        ModelLoader.set_detector(MockLoadedDetector(ai_probability=0.80))  # explain() returns None
        res_post = client.post(
            "/api/v1/analyze",
            files={"image": ("no_heat.jpg", sample_jpeg_bytes, "image/jpeg")}
        )
        analysis_id = res_post.json()["analysis_id"]
        assert res_post.json()["explanation"]["heatmap_available"] is False

        res_map = client.get(f"/api/v1/analyses/{analysis_id}/heatmap")
        assert res_map.status_code == 404
        assert res_map.json()["error"]["code"] == "HEATMAP_NOT_FOUND"
    finally:
        ModelLoader.set_detector(original_detector)


# =====================================================================
# 9. Existing heatmap returned correctly as image/jpeg
# =====================================================================
def test_frontend_existing_heatmap_returned_correctly(client, sample_jpeg_bytes):
    original_detector = ModelLoader.get_detector()
    try:
        ModelLoader.set_detector(MockDetectorWithHeatmap(ai_probability=0.95))
        res_post = client.post(
            "/api/v1/analyze",
            files={"image": ("with_heat.jpg", sample_jpeg_bytes, "image/jpeg")}
        )
        assert res_post.status_code == 200
        post_data = res_post.json()
        analysis_id = post_data["analysis_id"]

        assert post_data["explanation"]["heatmap_available"] is True
        assert post_data["explanation"]["heatmap_url"] == f"/api/v1/analyses/{analysis_id}/heatmap"

        # Fetch heatmap binary file stream
        res_map = client.get(f"/api/v1/analyses/{analysis_id}/heatmap")
        assert res_map.status_code == 200
        assert "image/jpeg" in res_map.headers.get("content-type", "")
        # Verify valid JPEG binary header (\xff\xd8\xff)
        assert res_map.content[:3] == b"\xff\xd8\xff"
    finally:
        ModelLoader.set_detector(original_detector)


# =====================================================================
# 10. All documented errors use canonical envelope
# =====================================================================
def test_frontend_canonical_error_envelope_across_endpoints(client):
    endpoints_to_test = [
        # (method, path, expected_status, expected_code)
        ("get", "/api/v1/analyses/not-a-valid-uuid", 400, "INVALID_ANALYSIS_ID"),
        ("get", f"/api/v1/analyses/{uuid.uuid4()}", 404, "ANALYSIS_NOT_FOUND"),
        ("get", f"/api/v1/analyses/{uuid.uuid4()}/heatmap", 404, "ANALYSIS_NOT_FOUND"),
        ("get", "/api/v1/analyses?page=-1", 400, "VALIDATION_ERROR"),
        ("get", "/api/v1/unknown_route", 404, "NOT_FOUND"),
        ("post", "/api/v1/health", 405, "METHOD_NOT_ALLOWED"),
    ]

    for method, path, expected_status, expected_code in endpoints_to_test:
        call = getattr(client, method)
        res = call(path)
        assert res.status_code == expected_status
        data = res.json()
        assert "error" in data, f"Missing 'error' wrapper in {path}"
        assert data["error"]["code"] == expected_code
        assert isinstance(data["error"]["message"], str)


# =====================================================================
# 11. No filesystem path in any response
# =====================================================================
def test_frontend_no_filesystem_paths_exposed(client, sample_jpeg_bytes):
    original_detector = ModelLoader.get_detector()
    try:
        ModelLoader.set_detector(MockLoadedDetector(0.8))
        res_post = client.post(
            "/api/v1/analyze",
            files={"image": ("path_check.jpg", sample_jpeg_bytes, "image/jpeg")}
        )
        assert res_post.status_code == 200
        text = res_post.text.lower()
        base_dir_str = str(settings.BASE_DIR).lower()
        assert base_dir_str not in text
        assert "c:\\" not in text
        assert "storage/originals" not in text

        # Error response path check
        res_err = client.get("/api/v1/analyses/bad-uuid")
        err_text = res_err.text.lower()
        assert base_dir_str not in err_text
        assert "c:\\" not in err_text
    finally:
        ModelLoader.set_detector(original_detector)


# =====================================================================
# 12. No traceback in any response
# =====================================================================
def test_frontend_no_traceback_exposed(client, sample_jpeg_bytes):
    original_detector = ModelLoader.get_detector()
    try:
        # Detector that raises an arbitrary Python internal exception
        class CrashDetector(ImageDetector):
            def is_loaded(self): return True
            def get_info(self): return {}
            def predict(self, t): raise ValueError("Unexpected vector matrix mismatch at line 42 in kernel.py")
            def explain(self, t): return None

        ModelLoader.set_detector(CrashDetector())
        res = client.post(
            "/api/v1/analyze",
            files={"image": ("crash.jpg", sample_jpeg_bytes, "image/jpeg")}
        )
        assert res.status_code == 500
        body = res.text
        assert "Traceback" not in body
        assert "kernel.py" not in body
        assert "matrix mismatch" not in body
        assert res.json()["error"]["code"] == "INFERENCE_ERROR"
    finally:
        ModelLoader.set_detector(original_detector)


# =====================================================================
# 13. Model unavailable remains 503
# =====================================================================
def test_frontend_model_unavailable_returns_503(client, sample_jpeg_bytes):
    ModelLoader._instance = None
    res = client.post(
        "/api/v1/analyze",
        files={"image": ("sample.jpg", sample_jpeg_bytes, "image/jpeg")}
    )
    assert res.status_code == 503
    data = res.json()
    assert data["error"]["code"] == "MODEL_UNAVAILABLE"
    assert "details" in data["error"]

