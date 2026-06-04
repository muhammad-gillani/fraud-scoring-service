"""
Phase 1 smoke tests — mock the model so tests run without a trained artifact.
"""

from unittest.mock import MagicMock, patch
import numpy as np
import pytest
from fastapi.testclient import TestClient


FRAUD_TRANSACTION = {
    "amount": 850.0,
    "hour": 2,
    "day_of_week": 5,
    "distance_from_home_km": 120.5,
    "distance_from_last_transaction_km": 80.0,
    "used_chip": 0,
    "used_pin": 0,
    "online_order": 1,
    "merchant_category": "electronics",
    "merchant_age_days": 45.0,
    "repeat_merchant": 0,
}

LEGIT_TRANSACTION = {
    **FRAUD_TRANSACTION,
    "amount": 12.0,
    "hour": 14,
    "distance_from_home_km": 2.0,
    "used_chip": 1,
    "used_pin": 1,
    "merchant_category": "grocery",
}

FEATURE_COLUMNS = [
    "amount", "hour", "day_of_week",
    "distance_from_home_km", "distance_from_last_transaction_km",
    "used_chip", "used_pin", "online_order",
    "merchant_age_days", "repeat_merchant",
    "merchant_category_electronics", "merchant_category_grocery",
    "merchant_category_gas", "merchant_category_restaurant",
    "merchant_category_online", "merchant_category_travel",
]


def make_mock_model(proba: float):
    mock = MagicMock()
    mock.predict_proba.return_value = np.array([[1 - proba, proba]])
    return mock


@pytest.fixture
def client():
    with patch("src.serving.api.load_artifacts") as mock_load:
        def side_effect():
            import src.serving.api as api
            api._model = make_mock_model(0.8)
            api._feature_columns = FEATURE_COLUMNS
        mock_load.side_effect = side_effect

        from src.serving.api import app
        with TestClient(app) as c:
            yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["model_loaded"] is True


def test_schema_returns_features(client):
    r = client.get("/schema")
    assert r.status_code == 200
    assert r.json()["n_features"] == len(FEATURE_COLUMNS)


def test_score_fraud(client):
    """High-risk transaction → is_fraud=True."""
    import src.serving.api as api
    api._model = make_mock_model(0.8)
    r = client.post("/score", json=FRAUD_TRANSACTION)
    assert r.status_code == 200
    data = r.json()
    assert data["is_fraud"] is True
    assert data["fraud_probability"] > 0.5


def test_missing_one_hot_filled(client):
    """Unknown merchant_category should not crash (reindex fills with 0)."""
    tx = {**FRAUD_TRANSACTION, "merchant_category": "unknown_new_category"}
    r = client.post("/score", json=tx)
    assert r.status_code == 200


def test_score_not_fraud(client):
    """Low-risk transaction → is_fraud=False."""
    import src.serving.api as api
    api._model = make_mock_model(0.05)
    r = client.post("/score", json=LEGIT_TRANSACTION)
    assert r.status_code == 200
    assert r.json()["is_fraud"] is False
