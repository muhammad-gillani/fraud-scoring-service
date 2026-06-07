"""
Train XGBoost fraud model and log everything to MLflow.
Saves model + feature schema locally for the serving API.
"""

import json
import os
from pathlib import Path

import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score
from xgboost import XGBClassifier

DATA_PATH = os.getenv("DATA_PATH", "data/raw/transactions.csv")
MODEL_DIR = os.getenv("MODEL_DIR", "models")
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
EXPERIMENT = "fraud-detection"

XGB_PARAMS = {
    "n_estimators": 300,
    "max_depth": 6,
    "learning_rate": 0.05,
    "scale_pos_weight": 10,  # handles class imbalance
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "eval_metric": "aucpr",
    "random_state": 42,
}


def load_data(path: str):
    df = pd.read_csv(path)
    y = df["is_fraud"]
    X = df.drop(columns=["is_fraud", "transaction_id"])
    X = pd.get_dummies(X, columns=["merchant_category"])
    return X, y


def plot_shap_summary(model, X_val: pd.DataFrame, output_path: str):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_val[:500])  # sample for speed
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.summary_plot(shap_values, X_val[:500], show=False)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def train():
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT)

    print(f"Loading data from {DATA_PATH}...")
    X, y = load_data(DATA_PATH)
    feature_columns = X.columns.tolist()

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    print(f"Train: {len(X_train):,} | Val: {len(X_val):,} | Fraud rate: {y.mean():.2%}")

    with mlflow.start_run():
        mlflow.log_params(XGB_PARAMS)
        mlflow.log_param("n_features", len(feature_columns))
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("val_size", len(X_val))

        model = XGBClassifier(**XGB_PARAMS)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=50)

        preds = model.predict_proba(X_val)[:, 1]
        roc_auc = roc_auc_score(y_val, preds)
        avg_precision = average_precision_score(y_val, preds)

        mlflow.log_metric("roc_auc", roc_auc)
        mlflow.log_metric("avg_precision", avg_precision)
        print(f"\nROC-AUC: {roc_auc:.4f} | Avg Precision: {avg_precision:.4f}")

        # SHAP artifact
        shap_path = "/tmp/shap_summary.png"
        plot_shap_summary(model, X_val, shap_path)
        mlflow.log_artifact(shap_path, artifact_path="plots")

        # Log feature schema as artifact (Phase 3: parity enforcement)
        schema_path = "/tmp/feature_columns.json"
        with open(schema_path, "w") as f:
            json.dump(feature_columns, f)
        mlflow.log_artifact(schema_path)

        # Log model
        # mlflow.xgboost.log_model(model, artifact_path="model") ## Phase 2: MLflow model registry
        mlflow.xgboost.log_model(
            model, 
            name="model", 
            registered_model_name="FraudModel")



        run_id = mlflow.active_run().info.run_id
        print(f"\nMLflow run_id: {run_id}")

    # Save locally for Docker volume mount
    Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)
    model_path = f"{MODEL_DIR}/fraud_model.json"
    schema_path = f"{MODEL_DIR}/feature_columns.json"
    model.save_model(model_path)
    with open(schema_path, "w") as f:
        json.dump(feature_columns, f)

    print(f"\nSaved model  → {model_path}")
    print(f"Saved schema → {schema_path}")
    print(f"\nMLflow UI    → {MLFLOW_URI}")


if __name__ == "__main__":
    train()
