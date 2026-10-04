"""
DealSight AI - Model Evaluation & Reporting Pipeline
Evaluates the best trained model on the hold-out test set, computes detailed metrics,
confusion matrix, and business-focused risk analysis.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

import sys
from pathlib import Path

# Add paths to sys.path
BACKEND_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = BACKEND_DIR.parent
for p in [str(BACKEND_DIR), str(PROJECT_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from ml.data_preprocessing import load_and_preprocess_data
except ImportError:
    from backend.ml.data_preprocessing import load_and_preprocess_data


def evaluate_best_model(model_path: str = None, data_path: str = None):
    """Evaluates the saved pipeline on the holdout test set."""
    if not model_path:
        candidates = [
            BACKEND_DIR / "models" / "deal_prediction_model.pkl",
            BACKEND_DIR / "models" / "best_model.pkl",
            PROJECT_ROOT / "models" / "deal_prediction_model.pkl",
            PROJECT_ROOT / "models" / "best_model.pkl",
        ]
        found = next((p for p in candidates if p.exists()), None)
        model_path = str(found) if found else str(BACKEND_DIR / "models" / "deal_prediction_model.pkl")

    if not data_path:
        data_path = str(
            BACKEND_DIR / "data" / "sales_data.csv"
            if (BACKEND_DIR / "data" / "sales_data.csv").exists()
            else PROJECT_ROOT / "data" / "sales_data.csv"
        )
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}. Run src/train_model.py first.")

    pipeline = joblib.load(model_path)
    X_train, X_val, X_test, y_train, y_val, y_test, _ = load_and_preprocess_data(data_path)

    # Inferences on holdout test set
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec_prog = recall_score(y_test, y_pred, zero_division=0)
    rec_lost = recall_score(y_test, y_pred, pos_label=0, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)

    tn, fp, fn, tp = cm.ravel()

    report = {
        "dataset_test_samples": len(y_test),
        "accuracy": round(float(acc), 4),
        "precision": round(float(prec), 4),
        "recall_progressed": round(float(rec_prog), 4),
        "recall_lost": round(float(rec_lost), 4),
        "f1_score": round(float(f1), 4),
        "roc_auc": round(float(roc_auc), 4),
        "confusion_matrix": {
            "true_negatives_lost": int(tn),
            "false_positives_predicted_progressed_but_lost": int(fp),
            "false_negatives_predicted_lost_but_progressed": int(fn),
            "true_positives_progressed": int(tp),
        },
        "business_summary": (
            f"The model successfully caught {rec_lost:.1%} of deals at risk of being Lost, "
            f"minimizing high-value deal pipeline slippage while preserving {prec:.1%} precision on progressed deals."
        )
    }

    print("\n" + "=" * 65)
    print("           DEALSIGHT AI - HOLDOUT TEST SET EVALUATION")
    print("=" * 65)
    print(f"Total Test Samples: {len(y_test)}")
    print(f"Accuracy:           {acc:.4f} ({acc*100:.1f}%)")
    print(f"Precision:          {prec:.4f}")
    print(f"Recall (Progress):  {rec_prog:.4f}")
    print(f"Recall (Lost/Risk): {rec_lost:.4f}  <-- Crucial risk metric")
    print(f"F1 Score:           {f1:.4f}")
    print(f"ROC-AUC:            {roc_auc:.4f}")
    print("-" * 65)
    print("Confusion Matrix:")
    print(f"  True Lost:        {tn:4d} | False Alarm (Pred Progress): {fp:4d}")
    print(f"  Missed (Pred Lost): {fn:4d} | True Progress:              {tp:4d}")
    print("-" * 65)
    print(report["business_summary"])
    print("=" * 65)

    target_save_path = BACKEND_DIR / "models" / "evaluation_report.json"
    with open(target_save_path, "w") as f:
        json.dump(report, f, indent=2)

    return report


if __name__ == "__main__":
    evaluate_best_model()
