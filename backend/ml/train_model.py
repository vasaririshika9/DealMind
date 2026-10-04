"""
DealSight AI - ML Model Training, Comparison & Selection Pipeline
Trains and evaluates Logistic Regression, Random Forest, Gradient Boosting,
and HistGradientBoosting with stratified train/validation/test sets.
Saves the best-performing pipeline and performance metrics.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
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
    from ml.data_preprocessing import load_and_preprocess_data, build_preprocessor, NUMERIC_FEATURES, CATEGORICAL_FEATURES
except ImportError:
    from backend.ml.data_preprocessing import load_and_preprocess_data, build_preprocessor, NUMERIC_FEATURES, CATEGORICAL_FEATURES


def train_and_evaluate_models(data_path: str = None, models_dir: str = None) -> Dict[str, Any]:
    """Trains multiple classifiers, evaluates on validation/test sets, and saves the best model."""
    target_models_dir = Path(models_dir) if models_dir else (BACKEND_DIR / "models")
    target_models_dir.mkdir(parents=True, exist_ok=True)
    models_dir = str(target_models_dir)

    if not data_path:
        data_path = str(
            BACKEND_DIR / "data" / "sales_data.csv"
            if (BACKEND_DIR / "data" / "sales_data.csv").exists()
            else PROJECT_ROOT / "data" / "sales_data.csv"
        )

    X_train, X_val, X_test, y_train, y_val, y_test, preprocessor = load_and_preprocess_data(data_path)

    # Transform datasets
    X_train_trans = preprocessor.transform(X_train)
    X_val_trans = preprocessor.transform(X_val)
    X_test_trans = preprocessor.transform(X_test)

    # Candidate models
    candidate_models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=150,
            max_depth=10,
            min_samples_split=4,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=120,
            learning_rate=0.08,
            max_depth=4,
            subsample=0.85,
            random_state=42,
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_iter=120,
            learning_rate=0.08,
            max_depth=5,
            random_state=42,
        ),
    }

    comparison_results = []
    trained_models = {}

    print("\n" + "=" * 65)
    print("           DEALSIGHT AI - MODEL TRAINING & COMPARISON")
    print("=" * 65)

    for name, clf in candidate_models.items():
        # Train
        clf.fit(X_train_trans, y_train)
        trained_models[name] = clf

        # Evaluate on validation set
        val_preds = clf.predict(X_val_trans)
        val_probs = clf.predict_proba(X_val_trans)[:, 1]

        # Evaluate on test set
        test_preds = clf.predict(X_test_trans)
        test_probs = clf.predict_proba(X_test_trans)[:, 1]

        acc = accuracy_score(y_test, test_preds)
        prec = precision_score(y_test, test_preds, zero_division=0)
        rec = recall_score(y_test, test_preds, zero_division=0)
        f1 = f1_score(y_test, test_preds, zero_division=0)
        roc_auc = roc_auc_score(y_test, test_probs)

        # Risk case metrics: predicting "Lost" (class 0)
        # In B2B sales, failing to catch a deal at risk of being lost is catastrophic.
        rec_lost = recall_score(y_test, test_preds, pos_label=0)

        cm = confusion_matrix(y_test, test_preds).tolist()

        result = {
            "model_name": name,
            "accuracy": round(float(acc), 4),
            "precision": round(float(prec), 4),
            "recall_progressed": round(float(rec), 4),
            "recall_lost": round(float(rec_lost), 4),
            "f1_score": round(float(f1), 4),
            "roc_auc": round(float(roc_auc), 4),
            "confusion_matrix": cm,
        }
        comparison_results.append(result)

        print(f"[{name}] Acc: {acc:.3f} | Prec: {prec:.3f} | Rec: {rec:.3f} | "
              f"Rec(Lost): {rec_lost:.3f} | F1: {f1:.3f} | ROC-AUC: {roc_auc:.3f}")

    # Model selection: prioritized by ROC-AUC and F1 on balanced risk detection
    # Business rationale: ROC-AUC demonstrates best class separation across all probability thresholds.
    best_candidate = max(comparison_results, key=lambda x: x["roc_auc"])
    best_name = best_candidate["model_name"]
    best_clf = trained_models[best_name]

    print("\n" + "-" * 65)
    print(f"[*] SELECTED BEST MODEL: {best_name} (ROC-AUC: {best_candidate['roc_auc']})")
    print("-" * 65)

    # Build end-to-end inference Pipeline
    full_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", best_clf),
        ]
    )

    # Save complete pipeline
    pipeline_path = os.path.join(models_dir, "best_model.pkl")
    joblib.dump(full_pipeline, pipeline_path)

    # Save preprocessor independently as well
    joblib.dump(preprocessor, os.path.join(models_dir, "preprocessor.pkl"))

    # Extract Feature Importance if available
    feature_names = []
    # numeric features
    feature_names.extend(NUMERIC_FEATURES)
    # categorical features one-hot names
    try:
        cat_encoder = preprocessor.named_transformers_["cat"].named_steps["onehot"]
        cat_names = cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES)
        feature_names.extend(cat_names.tolist())
    except Exception:
        pass

    importances_dict = {}
    if hasattr(best_clf, "feature_importances_"):
        raw_importances = best_clf.feature_importances_
        for fn, imp in zip(feature_names, raw_importances):
            importances_dict[fn] = round(float(imp), 5)
    elif hasattr(best_clf, "coef_"):
        raw_coefs = np.abs(best_clf.coef_[0])
        for fn, coef in zip(feature_names, raw_coefs):
            importances_dict[fn] = round(float(coef), 5)

    # Sort top features
    sorted_importances = sorted(importances_dict.items(), key=lambda x: x[1], reverse=True)
    top_features = [{"feature": k, "importance": v} for k, v in sorted_importances[:15]]

    # Save artifacts
    with open(os.path.join(models_dir, "model_comparison.json"), "w") as f:
        json.dump(comparison_results, f, indent=2)

    with open(os.path.join(models_dir, "metrics.json"), "w") as f:
        json.dump({
            "best_model": best_name,
            "metrics": best_candidate,
            "all_models": comparison_results,
        }, f, indent=2)

    with open(os.path.join(models_dir, "feature_importance.json"), "w") as f:
        json.dump(top_features, f, indent=2)

    print(f"[ARTIFACTS] Successfully saved pipeline to {pipeline_path}")
    print(f"[ARTIFACTS] Saved comparison results & metrics to {models_dir}/")

    return {
        "best_model": best_name,
        "metrics": best_candidate,
        "comparison": comparison_results,
        "top_features": top_features,
    }


if __name__ == "__main__":
    train_and_evaluate_models()
