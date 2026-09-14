"""
Integration Tests for API Router.
Verifies endpoint responses, status codes, and JSON schemas.
Responsible Team Member: Member 6 (MLOps & Testing)
"""

def test_api_health():
    from api.main import app
    from fastapi.testclient import TestClient
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
