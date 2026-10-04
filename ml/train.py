"""
DealSight AI - ML Model Training, Comparison & Selection Pipeline
File: ml/train.py

Trains and evaluates:
1. Logistic Regression
2. Random Forest
3. XGBoost (with HistGradientBoosting fallback)

Uses stratified 70% Train, 15% Validation, and 15% Test split with fixed random_state.
Calculates Accuracy, Precision, Recall (Progressed & Lost), F1, ROC-AUC, and Confusion Matrix.
Saves comparison to ml/model_results.json, ml/model_results.csv, and models/deal_prediction_model.pkl.
"""

import os
import sys
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

# Optional XGBoost import with safe fallback
try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except Exception as xgb_err:
    print(f"[NOTICE] XGBoost import notice: {xgb_err}. Will use HistGradientBoostingClassifier.")
    HAS_XGBOOST = False

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_preprocessing import (
    load_and_preprocess_data,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    ALL_PREDICTOR_FEATURES,
)


def train_models(
    data_path: str = "data/sales_data.csv",
    output_dir: str = "ml",
    models_dir: str = "models",
    random_state: int = 42,
) -> Dict[str, Any]:
    """Trains candidate classifiers, compares metrics, and serializes the best model."""
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    csv_full_path = PROJECT_ROOT / data_path
    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        preprocessor,
    ) = load_and_preprocess_data(str(csv_full_path), random_state=random_state)

    # Preprocess feature matrices
    X_train_trans = preprocessor.transform(X_train)
    X_val_trans = preprocessor.transform(X_val)
    X_test_trans = preprocessor.transform(X_test)

    # Define candidate models
    candidate_models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=random_state,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=150,
            max_depth=10,
            min_samples_split=4,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        ),
    }

    if HAS_XGBOOST:
        # Calculate scale_pos_weight for XGBoost to handle class balance
        neg_count = int((y_train == 0).sum())
        pos_count = int((y_train == 1).sum())
        scale_pos = neg_count / max(pos_count, 1)
        candidate_models["XGBoost"] = XGBClassifier(
            n_estimators=120,
            learning_rate=0.08,
            max_depth=4,
            subsample=0.85,
            scale_pos_weight=scale_pos,
            random_state=random_state,
            eval_metric="logloss",
        )
    else:
        candidate_models["HistGradientBoosting"] = HistGradientBoostingClassifier(
            max_iter=120,
            learning_rate=0.08,
            max_depth=5,
            random_state=random_state,
        )

    print("\n" + "=" * 70)
    print("      DEALSIGHT AI - MACHINE LEARNING MODEL TRAINING & COMPARISON")
    print("=" * 70)

    comparison_results: List[Dict[str, Any]] = []
    trained_models: Dict[str, Any] = {}

    for name, clf in candidate_models.items():
        print(f"\n[*] Training {name}...")
        clf.fit(X_train_trans, y_train)
        trained_models[name] = clf

        # Evaluate on Validation Set
        val_preds = clf.predict(X_val_trans)
        val_probs = clf.predict_proba(X_val_trans)[:, 1]
        val_acc = accuracy_score(y_val, val_preds)
        val_auc = roc_auc_score(y_val, val_probs)

        # Evaluate on Unseen Test Set
        test_preds = clf.predict(X_test_trans)
        test_probs = clf.predict_proba(X_test_trans)[:, 1]

        acc = accuracy_score(y_test, test_preds)
        prec = precision_score(y_test, test_preds, zero_division=0)
        rec = recall_score(y_test, test_preds, zero_division=0)
        rec_lost = recall_score(y_test, test_preds, pos_label=0)
        f1 = f1_score(y_test, test_preds, zero_division=0)
        roc_auc = roc_auc_score(y_test, test_probs)
        cm = confusion_matrix(y_test, test_preds).tolist()

        res_entry = {
            "model_name": name,
            "accuracy": round(float(acc), 4),
            "precision": round(float(prec), 4),
            "recall": round(float(rec), 4),
            "recall_lost": round(float(rec_lost), 4),
            "f1": round(float(f1), 4),
            "roc_auc": round(float(roc_auc), 4),
            "val_accuracy": round(float(val_acc), 4),
            "val_roc_auc": round(float(val_auc), 4),
            "confusion_matrix": cm,
        }
        comparison_results.append(res_entry)

        print(
            f"   Test Acc: {acc:.3f} | Prec: {prec:.3f} | Rec: {rec:.3f} | "
            f"Rec(Lost): {rec_lost:.3f} | F1: {f1:.3f} | ROC-AUC: {roc_auc:.3f}"
        )

    # Business Model Selection Logic:
    # In B2B sales forecasting, missing an at-risk deal (false negative on lost) costs thousands of dollars.
    # Therefore, we prioritize high Recall for Lost deals combined with high ROC-AUC.
    def selection_score(item: Dict[str, Any]) -> float:
        return item["roc_auc"] * 0.5 + item["recall_lost"] * 0.3 + item["f1"] * 0.2

    best_entry = max(comparison_results, key=selection_score)
    best_name = best_entry["model_name"]
    best_clf = trained_models[best_name]

    print("\n" + "-" * 70)
    print(f"[*] SELECTED BEST MODEL: {best_name}")
    print(
        f"    ROC-AUC: {best_entry['roc_auc']:.4f} | Recall on Lost Deals: {best_entry['recall_lost']:.4f} | F1: {best_entry['f1']:.4f}"
    )
    print(
        f"    Selection Rationale: {best_name} provided the best trade-off between identifying risky deals "
        f"(Recall on Lost: {best_entry['recall_lost']:.1%}) and overall class separation (ROC-AUC: {best_entry['roc_auc']:.3f})."
    )
    print("-" * 70)

    # Create end-to-end sklearn Pipeline with preprocessor and best classifier
    full_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", best_clf),
        ]
    )

    # Save Pipeline and Preprocessor
    model_save_path = PROJECT_ROOT / models_dir / "deal_prediction_model.pkl"
    prep_save_path = PROJECT_ROOT / models_dir / "preprocessing.pkl"
    joblib.dump(full_pipeline, model_save_path)
    joblib.dump(preprocessor, prep_save_path)

    # Maintain backwards compatibility aliases
    joblib.dump(full_pipeline, PROJECT_ROOT / models_dir / "best_model.pkl")
    joblib.dump(preprocessor, PROJECT_ROOT / models_dir / "preprocessor.pkl")

    # Save Model Comparison to CSV and JSON in ml/
    results_df = pd.DataFrame(comparison_results)
    results_csv_path = PROJECT_ROOT / output_dir / "model_results.csv"
    results_json_path = PROJECT_ROOT / output_dir / "model_results.json"
    results_df.to_csv(results_csv_path, index=False)

    with open(results_json_path, "w", encoding="utf-8") as f:
        json.dump(comparison_results, f, indent=2)

    # Save metadata
    feature_names: List[str] = list(NUMERIC_FEATURES)
    try:
        cat_encoder = preprocessor.named_transformers_["cat"].named_steps["onehot"]
        cat_names = cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES)
        feature_names.extend(cat_names.tolist())
    except Exception:
        pass

    metadata = {
        "model_version": "1.0.0",
        "training_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "selected_model": best_name,
        "evaluation_metrics": best_entry,
        "all_models_compared": comparison_results,
        "feature_count": len(feature_names),
        "features": feature_names,
        "sample_split": {
            "train_samples": len(X_train),
            "val_samples": len(X_val),
            "test_samples": len(X_test),
        },
    }

    metadata_path = PROJECT_ROOT / models_dir / "model_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    # Also update models/metrics.json
    with open(PROJECT_ROOT / models_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump({
            "best_model": best_name,
            "metrics": best_entry,
            "all_models": comparison_results,
        }, f, indent=2)

    # Extract Feature Importance if available
    importances_dict = {}
    if hasattr(best_clf, "feature_importances_"):
        raw_importances = best_clf.feature_importances_
        for fn, imp in zip(feature_names, raw_importances):
            importances_dict[fn] = round(float(imp), 5)
    elif hasattr(best_clf, "coef_"):
        raw_coefs = np.abs(best_clf.coef_[0])
        for fn, coef in zip(feature_names, raw_coefs):
            importances_dict[fn] = round(float(coef), 5)

    sorted_importances = sorted(importances_dict.items(), key=lambda x: x[1], reverse=True)
    top_features = [{"feature": k, "importance": v} for k, v in sorted_importances[:15]]
    with open(PROJECT_ROOT / models_dir / "feature_importance.json", "w", encoding="utf-8") as f:
        json.dump(top_features, f, indent=2)

    print(f"\n[ARTIFACTS] Saved model to: {model_save_path}")
    print(f"[ARTIFACTS] Saved results to: {results_csv_path} and {results_json_path}")
    print(f"[ARTIFACTS] Saved metadata to: {metadata_path}")

    return {
        "best_model": best_name,
        "metrics": best_entry,
        "all_models": comparison_results,
    }


if __name__ == "__main__":
    train_models()
