"""
Model evaluation report — generates plots and logs them to MLflow.
Run automatically after training, or standalone:
  python -m src.training.evaluate --run-id <mlflow_run_id>
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score,
    confusion_matrix,
    roc_auc_score,
)
from xgboost import XGBClassifier

MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
DATA_PATH = os.getenv("DATA_PATH", "data/raw/transactions.csv")
SCHEMA_PATH = os.getenv("SCHEMA_PATH", "models/feature_columns.json")
THRESHOLD = float(os.getenv("FRAUD_THRESHOLD", "0.5"))


def load_val_data():
    df = pd.read_csv(DATA_PATH)
    y = df["is_fraud"]
    X = df.drop(columns=["is_fraud", "transaction_id"])
    X = pd.get_dummies(X, columns=["merchant_category"])
    with open(SCHEMA_PATH) as f:
        feature_columns = json.load(f)
    X = X.reindex(columns=feature_columns, fill_value=0)
    # same split as training — reproducible
    from sklearn.model_selection import train_test_split
    _, X_val, _, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    return X_val, y_val

def load_train_data():
    df = pd.read_csv(DATA_PATH)
    y = df["is_fraud"]
    X = df.drop(columns=["is_fraud", "transaction_id"])
    X = pd.get_dummies(X, columns=["merchant_category"])
    with open(SCHEMA_PATH) as f:
        feature_columns = json.load(f)
    X = X.reindex(columns=feature_columns, fill_value=0)
    from sklearn.model_selection import train_test_split
    X_train, _, _, _ = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    return X_train, y

def plot_precision_recall(y_true, y_proba, ax):
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    ap = average_precision_score(y_true, y_proba)
    ax.plot(recall, precision, color="#7C3AED", linewidth=2)
    ax.fill_between(recall, precision, alpha=0.1, color="#7C3AED")
    ax.axvline(x=0.8, color="#D85A30", linestyle="--", linewidth=1, label="80% recall")
    ax.set_xlabel("Recall", fontsize=11)
    ax.set_ylabel("Precision", fontsize=11)
    ax.set_title(f"Precision-Recall Curve  (AP = {ap:.4f})", fontsize=12, fontweight="bold")
    ax.legend()
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    ax.grid(True, alpha=0.3)
    return precision, recall, thresholds


def plot_score_distribution(y_true, y_proba, ax):
    fraud_scores = y_proba[y_true == 1]
    legit_scores = y_proba[y_true == 0]
    bins = np.linspace(0, 1, 50)
    ax.hist(legit_scores, bins=bins, alpha=0.6, color="#0D9488", label=f"Legit (n={len(legit_scores):,})", density=True)
    ax.hist(fraud_scores, bins=bins, alpha=0.6, color="#D85A30", label=f"Fraud (n={len(fraud_scores):,})", density=True)
    ax.axvline(x=THRESHOLD, color="black", linestyle="--", linewidth=1.5, label=f"Threshold = {THRESHOLD}")
    ax.set_xlabel("Fraud Probability Score", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title("Score Distribution — Fraud vs Legit", fontsize=12, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)


def plot_confusion_matrix(y_true, y_proba, ax, threshold=0.5):
    y_pred = (y_proba >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    matrix = np.array([[tn, fp], [fn, tp]])
    labels = np.array([
        [f"True Neg\n{tn:,}", f"False Pos\n{fp:,}"],
        [f"False Neg\n{fn:,}", f"True Pos\n{tp:,}"]
    ])
    colors = np.array([[0.2, 0.6], [0.6, 0.2]])  # darker = worse

    im = ax.imshow(colors, cmap="RdYlGn", vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, labels[i, j], ha="center", va="center",
                   fontsize=12, fontweight="bold", color="black")

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Predicted Legit", "Predicted Fraud"], fontsize=10)
    ax.set_yticklabels(["Actual Legit", "Actual Fraud"], fontsize=10)
    ax.set_title(f"Confusion Matrix  (threshold = {threshold})", fontsize=12, fontweight="bold")

    fraud_catch_rate = tp / (tp + fn) if (tp + fn) > 0 else 0
    precision_val = tp / (tp + fp) if (tp + fp) > 0 else 0
    ax.set_xlabel(f"Catch rate (Recall): {fraud_catch_rate:.1%}  |  Precision: {precision_val:.1%}", fontsize=10)

def plot_shap_summary(model, X_train, X_val, output_dir="/tmp"):
    import shap
    explainer = shap.TreeExplainer(model)

    shap_train = explainer.shap_values(X_train.iloc[:1000])
    shap_val = explainer.shap_values(X_val.iloc[:500])

    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    fig.suptitle("SHAP Feature Importance — Train vs Validation", fontsize=14, fontweight="bold")

    plt.sca(axes[0])
    shap.summary_plot(shap_train, X_train.iloc[:1000], show=False, plot_size=None)
    axes[0].set_title("Training Set (n=1,000)", fontsize=12, fontweight="bold")

    plt.sca(axes[1])
    shap.summary_plot(shap_val, X_val.iloc[:500], show=False, plot_size=None)
    axes[1].set_title("Validation Set (n=500)", fontsize=12, fontweight="bold")

    plt.tight_layout()
    path = f"{output_dir}/shap_train_vs_val.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"SHAP comparison saved → {path}")
    return path

def plot_feature_importance(model, feature_columns, output_dir="/tmp"):
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    fig.suptitle("Feature Importance — Gain vs Weight", fontsize=14, fontweight="bold")

    importance_types = [("gain", "Total Gain", "#7C3AED"), ("weight", "Weight (Split Count)", "#0D9488")]

    for ax, (imp_type, title, color) in zip(axes, importance_types):
        scores = model.get_booster().get_score(importance_type=imp_type)
        # align to known feature columns, fill missing with 0
        scores_aligned = {f: scores.get(f, 0) for f in feature_columns}
        sorted_items = sorted(scores_aligned.items(), key=lambda x: x[1], reverse=False)
        features, values = zip(*sorted_items)

        ax.barh(features, values, color=color, alpha=0.8)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("Importance Score", fontsize=10)
        ax.grid(True, alpha=0.3, axis="x")
        ax.tick_params(axis="y", labelsize=8)

    plt.tight_layout()
    path = f"{output_dir}/feature_importance.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Feature importance saved → {path}")
    return path

def generate_report(run_id: str):
    mlflow.set_tracking_uri(MLFLOW_URI)

    print(f"Loading model from run: {run_id}")
    model = mlflow.xgboost.load_model(f"runs:/{run_id}/model")

    print("Loading validation data...")
    X_val, y_val = load_val_data()
    X_train, _ = load_train_data()
    y_proba = model.predict_proba(X_val)[:, 1]

    # Build report figure
    fig = plt.figure(figsize=(16, 5))
    fig.suptitle("Fraud Model — Evaluation Report", fontsize=14, fontweight="bold", y=1.02)
    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)

    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])

    plot_precision_recall(y_val.values, y_proba, ax1)
    plot_score_distribution(y_val.values, y_proba, ax2)
    plot_confusion_matrix(y_val.values, y_proba, ax3, threshold=THRESHOLD)

    report_path = "/tmp/evaluation_report.png"
    plt.savefig(report_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Report saved to {report_path}")

    # Log to MLflow under the same run
    with mlflow.start_run(run_id=run_id):
        mlflow.log_artifact(report_path, artifact_path="evaluation")

        shap_path = plot_shap_summary(model, X_train, X_val)
        mlflow.log_artifact(shap_path, artifact_path="evaluation")

        importance_path = plot_feature_importance(model, X_val.columns.tolist())
        mlflow.log_artifact(importance_path, artifact_path="evaluation")

        # Also log threshold-specific metrics
        y_pred = (y_proba >= THRESHOLD).astype(int)
        cm = confusion_matrix(y_val.values, y_pred)
        tn, fp, fn, tp = cm.ravel()
        mlflow.log_metrics({
            "catch_rate": round(tp / (tp + fn), 4),
            "precision_at_threshold": round(tp / (tp + fp), 4),
            "false_positive_rate": round(fp / (fp + tn), 4),
        })

    print(f"Report logged to MLflow run: {run_id}")
    print(f"View at: {MLFLOW_URI}/#/experiments/1/runs/{run_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True, help="MLflow run ID to evaluate")
    args = parser.parse_args()
    generate_report(args.run_id)
