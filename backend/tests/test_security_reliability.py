import io
import os
import uuid
import pytest
from unittest.mock import MagicMock
from PIL import Image
from sqlalchemy.exc import SQLAlchemyError
from fastapi.testclient import TestClient

from app.core.config import settings
from app.core.errors import (
    AppException,
    InvalidAnalysisIdError,
    InvalidImageError,
    FileTooLargeError,
)
from app.db.database import get_db
from app.db.models import AnalysisRecord
from app.main import create_app
from app.ml.model_loader import ModelLoader
from app.services.image_service import ImageService
from tests.conftest import MockLoadedDetector


def _get_stored_originals_count():
    originals_dir = settings.BASE_DIR / settings.originals_storage_path
    if not originals_dir.exists():
        return 0
    return len([f for f in originals_dir.glob("*") if f.is_file() and f.name != ".gitkeep"])


# =====================================================================
# 1. Malicious relative filename (../../bad.jpg)
# =====================================================================
def test_malicious_relative_filename_sanitized(client, db_session, sample_jpeg_bytes):
    original_detector = ModelLoader.get_detector()
    try:
        ModelLoader.set_detector(MockLoadedDetector(0.85))
        response = client.post(
            "/api/v1/analyze",
            files={"image": ("../../bad.jpg", sample_jpeg_bytes, "image/jpeg")}
        )
        assert response.status_code == 200
        analysis_id = response.json()["analysis_id"]

        record = db_session.query(AnalysisRecord).filter(AnalysisRecord.id == analysis_id).first()
        assert record is not None
        assert ".." not in record.filename
        assert "/" not in record.filename
        assert record.filename == "bad.jpg"

        # Stored file on disk must be UUID-named only
        assert os.path.exists(settings.BASE_DIR / record.stored_path)
        assert "bad.jpg" not in record.stored_path
    finally:
        ModelLoader.set_detector(original_detector)


# =====================================================================
# 2. Malicious Windows path filename (C:\Windows\System32\evil.jpg)
# =====================================================================
def test_malicious_windows_path_filename_sanitized(client, db_session, sample_jpeg_bytes):
    original_detector = ModelLoader.get_detector()
    try:
        ModelLoader.set_detector(MockLoadedDetector(0.85))
        response = client.post(
            "/api/v1/analyze",
            files={"image": ("C:\\Windows\\System32\\evil.jpg", sample_jpeg_bytes, "image/jpeg")}
        )
        assert response.status_code == 200
        analysis_id = response.json()["analysis_id"]

        record = db_session.query(AnalysisRecord).filter(AnalysisRecord.id == analysis_id).first()
        assert record is not None
        assert "C:" not in record.filename
        assert "\\" not in record.filename
        assert record.filename == "evil.jpg"
    finally:
        ModelLoader.set_detector(original_detector)


# =====================================================================
# 3. Absolute path filename (/etc/passwd.jpg)
# =====================================================================
def test_absolute_path_filename_sanitized(client, db_session, sample_jpeg_bytes):
    original_detector = ModelLoader.get_detector()
    try:
        ModelLoader.set_detector(MockLoadedDetector(0.85))
        response = client.post(
            "/api/v1/analyze",
            files={"image": ("/etc/passwd.jpg", sample_jpeg_bytes, "image/jpeg")}
        )
        assert response.status_code == 200
        analysis_id = response.json()["analysis_id"]

        record = db_session.query(AnalysisRecord).filter(AnalysisRecord.id == analysis_id).first()
        assert record is not None
        assert "/" not in record.filename
        assert record.filename == "passwd.jpg"
    finally:
        ModelLoader.set_detector(original_detector)


# =====================================================================
# 4. Very long filename (500+ characters)
# =====================================================================
def test_very_long_filename_handled_without_overflow(client, db_session, sample_jpeg_bytes):
    original_detector = ModelLoader.get_detector()
    try:
        ModelLoader.set_detector(MockLoadedDetector(0.85))
        long_filename = "A" * 500 + ".jpg"
        response = client.post(
            "/api/v1/analyze",
            files={"image": (long_filename, sample_jpeg_bytes, "image/jpeg")}
        )
        assert response.status_code == 200
        analysis_id = response.json()["analysis_id"]

        record = db_session.query(AnalysisRecord).filter(AnalysisRecord.id == analysis_id).first()
        assert record is not None
        assert len(record.filename) <= 255
    finally:
        ModelLoader.set_detector(original_detector)


# =====================================================================
# 5. Unicode filename (日本語_画像.png, файл.png)
# =====================================================================
def test_unicode_filename_preserved_safely(client, db_session, sample_png_bytes):
    original_detector = ModelLoader.get_detector()
    try:
        ModelLoader.set_detector(MockLoadedDetector(0.85))
        unicode_name = "日本語_画像_файл.png"
        response = client.post(
            "/api/v1/analyze",
            files={"image": (unicode_name, sample_png_bytes, "image/png")}
        )
        assert response.status_code == 200
        analysis_id = response.json()["analysis_id"]

        record = db_session.query(AnalysisRecord).filter(AnalysisRecord.id == analysis_id).first()
        assert record is not None
        assert record.filename == unicode_name
    finally:
        ModelLoader.set_detector(original_detector)


# =====================================================================
# 6. Fake image content (text disguised as .jpg)
# =====================================================================
def test_fake_image_content_rejected(client, db_session):
    files_before = _get_stored_originals_count()
    records_before = db_session.query(AnalysisRecord).count()

    response = client.post(
        "/api/v1/analyze",
        files={"image": ("disguised.jpg", b"This is plain text pretending to be JPEG bytes", "image/jpeg")}
    )
    assert response.status_code == 400
    data = response.json()
    assert data["error"]["code"] == "INVALID_IMAGE"

    assert _get_stored_originals_count() == files_before
    assert db_session.query(AnalysisRecord).count() == records_before


# =====================================================================
# 7. Invalid UUID formats rejected with 400 INVALID_ANALYSIS_ID
# =====================================================================
@pytest.mark.parametrize("invalid_id", [
    "not-a-uuid",
    "12345",
    "../../etc/passwd",
    "'; DROP TABLE analyses; --",
    "00000000-0000-0000-0000-00000000000Z",
])
def test_invalid_uuid_rejected(client, invalid_id):
    from urllib.parse import quote
    encoded_id = quote(invalid_id, safe="")

    # Test GET /analyses/{id}
    res1 = client.get(f"/api/v1/analyses/{encoded_id}")
    assert res1.status_code == 400
    assert res1.json()["error"]["code"] == "INVALID_ANALYSIS_ID"

    # Test GET /analyses/{id}/heatmap
    res2 = client.get(f"/api/v1/analyses/{encoded_id}/heatmap")
    assert res2.status_code == 400
    assert res2.json()["error"]["code"] == "INVALID_ANALYSIS_ID"


# =====================================================================
# 8. Nonexistent valid UUID returns 404 ANALYSIS_NOT_FOUND
# =====================================================================
def test_nonexistent_valid_uuid_returns_404(client):
    nonexistent = str(uuid.uuid4())
    res1 = client.get(f"/api/v1/analyses/{nonexistent}")
    assert res1.status_code == 404
    assert res1.json()["error"]["code"] == "ANALYSIS_NOT_FOUND"

    res2 = client.get(f"/api/v1/analyses/{nonexistent}/heatmap")
    assert res2.status_code == 404
    assert res2.json()["error"]["code"] == "ANALYSIS_NOT_FOUND"


# =====================================================================
# 9. Pagination below valid range (page=0, page=-1)
# =====================================================================
@pytest.mark.parametrize("invalid_page", [0, -1, -99])
def test_pagination_below_valid_range_rejected(client, invalid_page):
    res = client.get(f"/api/v1/analyses?page={invalid_page}")
    assert res.status_code == 400
    data = res.json()
    assert data["error"]["code"] == "VALIDATION_ERROR"


# =====================================================================
# 10. Pagination above valid range (page_size=101, page_size=9999)
# =====================================================================
@pytest.mark.parametrize("invalid_size", [101, 500, 9999])
def test_pagination_above_valid_range_rejected(client, invalid_size):
    res = client.get(f"/api/v1/analyses?page_size={invalid_size}")
    assert res.status_code == 400
    data = res.json()
    assert data["error"]["code"] == "VALIDATION_ERROR"


# =====================================================================
# 11. Oversized upload rejected (413 FILE_TOO_LARGE)
# =====================================================================
def test_oversized_upload_rejected_via_api(client, monkeypatch, db_session):
    monkeypatch.setattr(settings, "MAX_UPLOAD_SIZE_BYTES", 1024)
    oversized_payload = b"\xff\xd8\xff" + (b"\x00" * 2048)

    files_before = _get_stored_originals_count()
    records_before = db_session.query(AnalysisRecord).count()

    res = client.post(
        "/api/v1/analyze",
        files={"image": ("too_big.jpg", oversized_payload, "image/jpeg")}
    )
    assert res.status_code == 413
    assert res.json()["error"]["code"] == "FILE_TOO_LARGE"

    assert _get_stored_originals_count() == files_before
    assert db_session.query(AnalysisRecord).count() == records_before


# =====================================================================
# 12. Client error messages contain no server paths
# =====================================================================
def test_client_error_messages_contain_no_server_paths(client, sample_jpeg_bytes):
    base_dir_lower = str(settings.BASE_DIR).lower()

    # 1. 400 Invalid Image
    r1 = client.post("/api/v1/analyze", files={"image": ("bad.jpg", b"fake", "image/jpeg")})
    assert base_dir_lower not in r1.text.lower()
    assert "storage/originals" not in r1.text.lower()
    assert "c:\\" not in r1.text.lower()

    # 2. 400 Invalid UUID
    r2 = client.get("/api/v1/analyses/invalid-uuid-string")
    assert base_dir_lower not in r2.text.lower()
    assert "c:\\" not in r2.text.lower()

    # 3. 404 Nonexistent UUID
    r3 = client.get(f"/api/v1/analyses/{uuid.uuid4()}")
    assert base_dir_lower not in r3.text.lower()
    assert "c:\\" not in r3.text.lower()

    # 4. 503 Model Unavailable
    ModelLoader._instance = None
    r4 = client.post("/api/v1/analyze", files={"image": ("test.jpg", sample_jpeg_bytes, "image/jpeg")})
    assert base_dir_lower not in r4.text.lower()
    assert "c:\\" not in r4.text.lower()


# =====================================================================
# 13. Client error messages contain no traceback
# =====================================================================
def test_client_error_messages_contain_no_traceback(client, sample_jpeg_bytes):
    original_detector = ModelLoader.get_detector()
    try:
        mock_broken = MagicMock()
        mock_broken.is_loaded.return_value = True
        mock_broken.predict.side_effect = RuntimeError("Internal neural computation crashed unexpectedly")
        ModelLoader.set_detector(mock_broken)

        res = client.post(
            "/api/v1/analyze",
            files={"image": ("test.jpg", sample_jpeg_bytes, "image/jpeg")}
        )
        assert res.status_code == 500
        body = res.text
        assert "Traceback" not in body
        assert "line " not in body
        assert ".py" not in body
        assert "RuntimeError" not in body
        assert "neural computation" not in body
        assert res.json()["error"]["code"] in ("INFERENCE_ERROR", "INTERNAL_SERVER_ERROR")
    finally:
        ModelLoader.set_detector(original_detector)


# =====================================================================
# 14. Storage failure cleanup (simulated disk write error)
# =====================================================================
def test_storage_failure_cleans_up_and_rolls_back(client, db_session, sample_jpeg_bytes, monkeypatch):
    original_detector = ModelLoader.get_detector()
    try:
        ModelLoader.set_detector(MockLoadedDetector(0.85))
        files_before = _get_stored_originals_count()
        records_before = db_session.query(AnalysisRecord).count()

        # Simulate disk write failure in save_original
        def mock_save_original(*args, **kwargs):
            raise OSError("Simulated disk full / permission error")

        from app.services.storage_service import StorageService
        monkeypatch.setattr(StorageService, "save_original", mock_save_original)

        res = client.post(
            "/api/v1/analyze",
            files={"image": ("test.jpg", sample_jpeg_bytes, "image/jpeg")}
        )
        assert res.status_code == 500
        assert res.json()["error"]["code"] == "ANALYSIS_FAILED"

        # Verify zero side effects
        assert _get_stored_originals_count() == files_before
        assert db_session.query(AnalysisRecord).count() == records_before
    finally:
        ModelLoader.set_detector(original_detector)


# =====================================================================
# 15. Database failure cleanup (simulated DB commit error)
# =====================================================================
def test_database_failure_cleans_up_stored_image(client, db_session, sample_jpeg_bytes, monkeypatch):
    original_detector = ModelLoader.get_detector()
    try:
        ModelLoader.set_detector(MockLoadedDetector(0.85))
        files_before = _get_stored_originals_count()
        records_before = db_session.query(AnalysisRecord).count()

        # Simulate failure during db.commit()
        real_commit = db_session.commit

        def failing_commit():
            raise SQLAlchemyError("Simulated database disk I/O lock failure")

        monkeypatch.setattr(db_session, "commit", failing_commit)

        res = client.post(
            "/api/v1/analyze",
            files={"image": ("test.jpg", sample_jpeg_bytes, "image/jpeg")}
        )
        assert res.status_code == 500
        assert res.json()["error"]["code"] == "DATABASE_ERROR"

        # Verify that newly saved original file was deleted by transactional cleanup
        assert _get_stored_originals_count() == files_before
        assert db_session.query(AnalysisRecord).count() == records_before
    finally:
        ModelLoader.set_detector(original_detector)


# =====================================================================
# 16. CORS configuration behavior
# =====================================================================
def test_cors_configuration_origin_handling(monkeypatch):
    # Set CORS_ORIGINS to a specific domain
    monkeypatch.setattr(settings, "CORS_ORIGINS", ["https://trusted-portal.org"])
    app = create_app()

    with TestClient(app) as test_client:
        # 1. Request from trusted origin
        res_trusted = test_client.get(
            "/api/v1/health",
            headers={"Origin": "https://trusted-portal.org"}
        )
        assert res_trusted.status_code == 200
        assert res_trusted.headers.get("access-control-allow-origin") == "https://trusted-portal.org"

        # 2. Request from untrusted origin
        res_untrusted = test_client.get(
            "/api/v1/health",
            headers={"Origin": "https://attacker-domain.com"}
        )
        assert res_untrusted.status_code == 200
        assert res_untrusted.headers.get("access-control-allow-origin") is None


# =====================================================================
# Decompression bomb / pathological pixel dimensions protection
# =====================================================================
def test_decompression_bomb_protection(monkeypatch):
    # Set a tiny max pixels threshold to simulate a decompression bomb
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 100)

    img = Image.new("RGB", (64, 64), color=(100, 100, 100))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    payload = buf.getvalue()

    with pytest.raises(InvalidImageError) as exc_info:
        ImageService.validate_and_decode(payload)

    assert exc_info.value.code == "INVALID_IMAGE"
    assert "exceeds" in exc_info.value.message.lower()
