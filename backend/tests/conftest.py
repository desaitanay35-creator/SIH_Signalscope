import io
import sys
from pathlib import Path
import pytest
from PIL import Image
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

# Ensure backend root directory is in sys.path
backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app.db.database import Base, get_db
from app.main import create_app
from app.ml.detector import ImageDetector
from app.ml.model_loader import ModelLoader


class MockLoadedDetector(ImageDetector):
    """
    Test-only mock detector to verify end-to-end API pipeline serialization.
    Not used in production code.
    """
    def __init__(self, ai_probability: float = 0.88):
        self.ai_probability = ai_probability

    def is_loaded(self) -> bool:
        return True

    def get_info(self) -> dict:
        return {
            "name": "SignalScope Test Detector",
            "version": "test-v1",
            "task": "real-vs-ai-generated",
            "loaded": True,
            "device": "cpu"
        }

    def predict(self, image_tensor) -> dict:
        return {
            "raw_ai_probability": self.ai_probability,
            "raw_real_probability": 1.0 - self.ai_probability,
            "model_version": "test-v1"
        }

    def explain(self, image_tensor):
        return None


@pytest.fixture
def db_session():
    """In-memory SQLite session with StaticPool for fast, leak-free test isolation."""
    test_engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool
    )
    Base.metadata.create_all(bind=test_engine)
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(bind=test_engine)
        test_engine.dispose()


@pytest.fixture(autouse=True)
def clean_storage():
    """Ensures test storage directories are pristine before and after each test."""
    from app.core.config import settings

    def _purge():
        for sub in [settings.originals_storage_path, settings.heatmaps_storage_path]:
            target_dir = settings.BASE_DIR / sub
            if target_dir.exists():
                for f in target_dir.glob("*"):
                    if f.is_file() and f.name != ".gitkeep":
                        try:
                            f.unlink()
                        except Exception:
                            pass

    _purge()
    yield
    _purge()


@pytest.fixture
def client(db_session):
    app = create_app()

    def override_get_db():
        yield db_session

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


@pytest.fixture
def sample_jpeg_bytes() -> bytes:
    """Generates a minimal valid JPEG image in-memory."""
    img = Image.new("RGB", (128, 128), color=(70, 130, 180))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture
def sample_png_bytes() -> bytes:
    """Generates a minimal valid PNG image in-memory."""
    img = Image.new("RGB", (64, 64), color=(34, 139, 34))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def sample_webp_bytes() -> bytes:
    """Generates a minimal valid WEBP image in-memory."""
    img = Image.new("RGB", (64, 64), color=(255, 69, 0))
    buf = io.BytesIO()
    img.save(buf, format="WEBP")
    return buf.getvalue()
