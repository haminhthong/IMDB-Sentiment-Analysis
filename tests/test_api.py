"""Kiểm thử tự động cho FastAPI REST API trong api.py."""

import pytest

pytest.importorskip("torch")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient

import api
from api import app
from sentiment.inference import PredictionResult


class FakePredictor:
    """Predictor nhỏ để test API không phụ thuộc checkpoint ngoài repository."""

    def predict(self, text: str) -> PredictionResult:
        return PredictionResult(
            label="Positive",
            probability=0.75,
            token_count=len(text.split()),
        )

    def predict_batch(self, texts: list[str]) -> list[PredictionResult]:
        return [self.predict(text) for text in texts]


@pytest.fixture(autouse=True)
def mock_predictor(monkeypatch):
    """Thay predictor thật bằng fake để test chỉ tập trung vào HTTP contract."""
    monkeypatch.setattr(api, "get_predictor", lambda: FakePredictor())


client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "CineSentiment" in data["service"]


def test_predict_validation_error():
    # Gửi request rỗng
    response = client.post("/predict", json={"text": ""})
    assert response.status_code == 422  # Pydantic validation error


def test_predict_batch_validation_error():
    response = client.post("/predict/batch", json={"texts": []})
    assert response.status_code == 422


def test_predict_endpoint_success():
    response = client.post("/predict", json={"text": "A wonderful film with great acting."})
    assert response.status_code == 200
    data = response.json()
    assert "label" in data
    assert "probability" in data
    assert isinstance(data["probability"], float)


def test_predict_batch_endpoint_success():
    response = client.post(
        "/predict/batch",
        json={"texts": ["Great movie!", "Terrible and boring."]},
    )
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) == 2
    assert "label" in data[0]
