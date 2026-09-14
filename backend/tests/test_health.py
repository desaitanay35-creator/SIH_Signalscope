from app.ml.model_loader import ModelLoader
from tests.conftest import MockLoadedDetector


def test_health_endpoint_model_unloaded(client):
    """Verifies /api/v1/health returns ok and model_loaded=False when weights missing."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is False


def test_model_endpoint_unloaded(client):
    """Verifies /api/v1/model returns detector metadata with loaded=False."""
    response = client.get("/api/v1/model")
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "SignalScope Detector"
    assert "version" in data
    assert data["task"] == "real-vs-ai-generated"
    assert data["loaded"] is False


def test_health_endpoint_model_loaded(client):
    """Verifies /api/v1/health reflects loaded status when detector is loaded."""
    original_detector = ModelLoader.get_detector()
    try:
        ModelLoader.set_detector(MockLoadedDetector(0.9))
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        assert response.json()["model_loaded"] is True
    finally:
        ModelLoader.set_detector(original_detector)

