# Fraud Scoring Service

A production-grade real-time fraud scoring system built to demonstrate the full MLOps lifecycle — from training to serving, monitoring, and retraining. Built as a personal project to bridge Data Science → ML Engineering.

---

## Tech Stack

| Layer | Tools |
|---|---|
| Model | XGBoost, scikit-learn, SHAP |
| Experiment tracking | MLflow |
| Serving | FastAPI, Uvicorn |
| Containerisation | Docker, Docker Compose |
| Online features | Redis (Phase 3) |
| Monitoring | Evidently, Prometheus, Grafana (Phase 4) |
| Orchestration | Apache Airflow (Phase 5) |
| CI/CD | GitHub Actions (Phase 5) |

---

## Phases

### ✅ Phase 1 — Walking skeleton
Goal: one transaction in, fraud score out. End-to-end on day one.
- [x] Synthetic data generator (swap in real data anytime)
- [x] XGBoost training pipeline with MLflow experiment tracking
- [x] FastAPI `POST /score` inference endpoint
- [x] `/health` and `/schema` endpoints
- [x] Dockerised service with hot-mounted model volume
- [x] Smoke tests

### 🔲 Phase 2 — Training pipeline hardening
- [ ] `config.yaml` for all hyperparams and thresholds
- [ ] Precision-recall curves, score distribution plots
- [ ] Model promotion workflow (Staging → Production) in MLflow registry

### 🔲 Phase 3 — Online feature serving (signature piece)
- [ ] Offline feature computation script (batch, matches training)
- [ ] Online feature lookup via Redis
- [ ] `tests/test_parity.py` — asserts offline == online for same transaction
- [ ] `/schema` endpoint used by parity tests to enforce contract

> This phase directly replicates the online/offline score discrepancy problem
> solved in production at i2c. Building it yourself makes that interview story bulletproof.

### 🔲 Phase 4 — Monitoring
- [ ] Data drift detection with Evidently (score distribution, feature drift)
- [ ] Prometheus metrics instrumented in FastAPI
- [ ] Grafana dashboard (p50/p99 latency, fraud rate, drift alerts)

### 🔲 Phase 5 — Automation
- [ ] Airflow DAG for scheduled retraining + model promotion
- [ ] GitHub Actions CI/CD: test → build → push on every commit

### 🔲 Phase 6 — Stretch
- [ ] Cloud deployment (AWS free tier / Fly.io)
- [ ] Load testing with Locust
- [ ] LLM explanation layer: plain-English reason for each flagged transaction

---

## Quick Start

```bash
# 1. Install dependencies
make setup

# 2. Generate synthetic data (or drop real CSV into data/raw/)
make data

# 3. Start MLflow + API via Docker
make docker-up

# 4. Train and register the model
make train

# 5. Test a live score
make score
```

Open **MLflow UI** at http://localhost:5000 and **API docs** at http://localhost:8000/docs.

---

## Using Real Data

Download the [IEEE-CIS Fraud Detection](https://www.kaggle.com/competitions/ieee-fraud-detection/data) dataset from Kaggle.
Place `train_transaction.csv` in `data/raw/` and set:

```bash
export DATA_PATH=data/raw/train_transaction.csv
make train
```

---

## Project Structure

```
fraud-scoring-service/
├── src/
│   ├── data/generate_synthetic.py    # instant start, no Kaggle needed
│   ├── training/train.py             # XGBoost + MLflow (Phase 1)
│   └── serving/api.py                # FastAPI scoring endpoint (Phase 1)
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml            # API + MLflow
├── tests/
│   ├── test_api.py                   # Phase 1 smoke tests
│   └── test_parity.py                # Phase 3: offline == online assertion
├── models/                           # populated by training (gitignored)
├── data/raw/                         # datasets (gitignored)
├── requirements.txt
├── Makefile
└── README.md
```

---

## Interview Note: Feature Parity

Phase 3 of this project deliberately solves the online/offline score discrepancy problem — the same class of issue traced and fixed in production at i2c Inc. The pattern: features computed offline during training must exactly match features served online at inference. `tests/test_parity.py` enforces this as an automated contract. The `/schema` endpoint exposes the expected feature list so any serving layer can validate against it.
