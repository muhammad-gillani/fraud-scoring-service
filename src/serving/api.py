"""
Fraud Scoring Service — FastAPI app.
Loads XGBoost model at startup, serves scores via POST /score.
"""

import json
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from xgboost import XGBClassifier

import mlflow
import mlflow.xgboost

MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MODEL_NAME = os.getenv("MODEL_NAME", "FraudModel")
MODEL_ALIAS = os.getenv("MODEL_ALIAS", "production")
SCHEMA_PATH = os.getenv("SCHEMA_PATH", "models/feature_columns.json")
FRAUD_THRESHOLD = float(os.getenv("FRAUD_THRESHOLD", "0.5"))

# Global state — loaded once at startup
_model: XGBClassifier | None = None
_feature_columns: list[str] | None = None


def load_artifacts():
    global _model, _feature_columns
    mlflow.set_tracking_uri(MLFLOW_URI)
    model_uri = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
    _model = mlflow.xgboost.load_model(model_uri)
    with open(SCHEMA_PATH) as f:
        _feature_columns = json.load(f)
    print(f"Model loaded from registry: {model_uri} | {len(_feature_columns)} features | threshold={FRAUD_THRESHOLD}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_artifacts()
    yield
    # cleanup on shutdown (nothing needed for Phase 1)


app = FastAPI(
    title="Fraud Scoring Service",
    description="Real-time transaction fraud scoring. Phase 1 — XGBoost + MLflow.",
    version="0.1.0",
    lifespan=lifespan,
)


# ── Request / Response schemas ────────────────────────────────────────────────

class Transaction(BaseModel):
    amount: float
    hour: int
    day_of_week: int
    distance_from_home_km: float
    distance_from_last_transaction_km: float
    used_chip: int
    used_pin: int
    online_order: int
    merchant_category: str
    merchant_age_days: float
    repeat_merchant: int

    model_config = {"json_schema_extra": {"example": {
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
    }}}


class ScoreResponse(BaseModel):
    fraud_probability: float
    is_fraud: bool
    threshold: float
    latency_ms: float
    n_features_used: int


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _model is not None}


@app.get("/schema")
def schema():
    """Returns the feature contract. Phase 3 uses this to enforce parity."""
    if _feature_columns is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"features": _feature_columns, "n_features": len(_feature_columns)}


@app.post("/score", response_model=ScoreResponse)
def score(transaction: Transaction):
    if _model is None or _feature_columns is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    t0 = time.perf_counter()

    # Build raw feature dict then one-hot encode merchant_category
    raw = transaction.model_dump()
    df = pd.DataFrame([raw])
    df = pd.get_dummies(df, columns=["merchant_category"])

    # Align to training schema — fills missing one-hot columns with 0
    df = df.reindex(columns=_feature_columns, fill_value=0)

    proba = float(_model.predict_proba(df)[0, 1])
    latency_ms = (time.perf_counter() - t0) * 1000

    return ScoreResponse(
        fraud_probability=round(proba, 6),
        is_fraud=proba >= FRAUD_THRESHOLD,
        threshold=FRAUD_THRESHOLD,
        latency_ms=round(latency_ms, 3),
        n_features_used=len(_feature_columns),
    )
