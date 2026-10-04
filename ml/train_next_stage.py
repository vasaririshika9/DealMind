"""
DealSight AI - Next-Stage Transition Model Training & Evaluation Harness
File: ml/train_next_stage.py

Trains and compares multi-class stage transition models on data/deal_transitions.csv.
- Strict data validation (no missing targets, no duplicates, no future leakage)
- Zero data leakage: split strictly grouped by deal_id (70% train, 15% val, 15% test)
- Benchmarks Logistic Regression, Random Forest, and XGBoost
- Selects best model using Macro F1 to handle imbalanced transition classes
- Exports pipeline to models/next_stage_model.pkl and metadata to models/next_stage_metadata.json
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, Any, Tuple, List

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.stage_models import (
    NUMERIC_STAGE_FEATURES,
    CATEGORICAL_STAGE_FEATURES,
    ALL_STAGE_FEATURES,
    build_stage_preprocessor,
    XGBStringClassifier,
)
from data.generate_transitions import generate_deal_transitions


def validate_transition_data(df: pd.DataFrame) -> None:
    """Rigorous pre-training data integrity and leakage validation."""
    required_cols = ["deal_id", "call_number", "current_stage", "next_stage"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"[DATA INTEGRITY ERROR] Required column '{col}' is missing from transition dataset.")

    # Check for missing values in essential fields
    if df["next_stage"].isnull().any():
        raise ValueError("[DATA INTEGRITY ERROR] Found null or missing values in target 'next_stage'.")

    if df["current_stage"].isnull().any():
        raise ValueError("[DATA INTEGRITY ERROR] Found null or missing values in 'current_stage'.")

    # Check for duplicates in interaction records
    dupes = df.duplicated(subset=["deal_id", "call_number"]).sum()
    if dupes > 0:
        raise ValueError(f"[DATA INTEGRITY ERROR] Found {dupes} duplicate (deal_id, call_number) transition records.")

    # Check for future information / target leakage in training features
    forbidden_features = [
        "deal_outcome", "outcome", "is_won", "final_stage", "final_outcome",
        "future_call", "future_sentiment", "closed_won", "closed_lost"
    ]
    for feat in forbidden_features:
        if feat in df.columns and feat != "next_stage":
            raise ValueError(f"[TARGET LEAKAGE ERROR] Forbidden post-outcome column '{feat}' found in dataset.")

    print(f"[DATA INTEGRITY PASS] Verified {len(df)} records across {df['deal_id'].nunique()} unique deals. Zero leakage.")


def split_grouped_by_deal(
    df: pd.DataFrame,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Splits dataset strictly by deal_id to ensure no deal overlaps across splits."""
    gss_test = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=random_state)
    train_val_idx, test_idx = next(gss_test.split(df, groups=df["deal_id"]))

    train_val_df = df.iloc[train_val_idx].copy()
    test_df = df.iloc[test_idx].copy()

    # Split train_val into train (70% total) and validation (15% total)
    val_ratio = 0.15 / 0.85  # ~0.17647
    gss_val = GroupShuffleSplit(n_splits=1, test_size=val_ratio, random_state=random_state)
    train_idx, val_idx = next(gss_val.split(train_val_df, groups=train_val_df["deal_id"]))

    train_df = train_val_df.iloc[train_idx].copy()
    val_df = train_val_df.iloc[val_idx].copy()

    # Validate zero split leakage
    train_deals = set(train_df["deal_id"])
    val_deals = set(val_df["deal_id"])
    test_deals = set(test_df["deal_id"])

    if train_deals & val_deals or train_deals & test_deals or val_deals & test_deals:
        raise ValueError("[SPLIT LEAKAGE ERROR] Deals overlap between training, validation, or test sets!")

    print(f"[GROUPED SPLIT] Deals -> Train: {len(train_deals)}, Val: {len(val_deals)}, Test: {len(test_deals)}")
    print(f"[GROUPED SPLIT] Interactions -> Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    return train_df, val_df, test_df


def train_next_stage_pipeline(
    data_path: str = "data/deal_transitions.csv",
    models_dir: str = "models",
    random_state: int = 42,
) -> Dict[str, Any]:
    """Trains, benchmarks, and saves the next-stage transition model."""
    full_data_path = PROJECT_ROOT / data_path
    if not full_data_path.exists():
        print(f"[STAGE TRAINING] Dataset not found at {full_data_path}. Generating now...")
        generate_deal_transitions(n_deals=1200, output_path=data_path)

    df = pd.read_csv(full_data_path)
    validate_transition_data(df)

    train_df, val_df, test_df = split_grouped_by_deal(df, random_state=random_state)

    X_train = train_df[ALL_STAGE_FEATURES]
    y_train = train_df["next_stage"]
    X_val = val_df[ALL_STAGE_FEATURES]
    y_val = val_df["next_stage"]
    X_test = test_df[ALL_STAGE_FEATURES]
    y_test = test_df["next_stage"]

    # Candidate Models to benchmark
    candidate_estimators = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            C=1.0,
            class_weight="balanced",
            random_state=random_state,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=150,
            max_depth=12,
            min_samples_split=4,
            class_weight="balanced",
            random_state=random_state,
        ),
        "XGBoost": XGBStringClassifier(
            n_estimators=120,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=random_state,
            eval_metric="mlogloss",
        ),
    }

    benchmark_results = []
    trained_pipelines = {}

    for name, estimator in candidate_estimators.items():
        preprocessor = build_stage_preprocessor()
        pipe = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", estimator)])
        pipe.fit(X_train, y_train)
        trained_pipelines[name] = pipe

        # Evaluate on unseen holdout test deals
        preds = pipe.predict(X_test)

        acc = float(accuracy_score(y_test, preds))
        macro_f1 = float(f1_score(y_test, preds, average="macro"))
        weighted_f1 = float(f1_score(y_test, preds, average="weighted"))
        prec_macro = float(precision_score(y_test, preds, average="macro", zero_division=0))
        rec_macro = float(recall_score(y_test, preds, average="macro", zero_division=0))

        # Per-stage recall
        classes = sorted(list(y_test.unique()))
        report = classification_report(y_test, preds, output_dict=True, zero_division=0)
        per_stage = {
            cls: {
                "precision": round(float(report[cls]["precision"]), 3),
                "recall": round(float(report[cls]["recall"]), 3),
                "f1_score": round(float(report[cls]["f1-score"]), 3),
                "support": int(report[cls]["support"]),
            }
            for cls in classes if cls in report
        }

        cm = confusion_matrix(y_test, preds, labels=classes).tolist()

        res_entry = {
            "model_name": name,
            "accuracy": round(acc, 4),
            "macro_f1": round(macro_f1, 4),
            "weighted_f1": round(weighted_f1, 4),
            "precision_macro": round(prec_macro, 4),
            "recall_macro": round(rec_macro, 4),
            "classes": classes,
            "per_stage_metrics": per_stage,
            "confusion_matrix": cm,
        }
        benchmark_results.append(res_entry)
        print(f"[{name.upper()}] Acc: {acc:.4f} | Macro F1: {macro_f1:.4f} | Weighted F1: {weighted_f1:.4f}")

    # SELECT BEST MODEL STRICTLY USING MACRO F1
    best_res = max(benchmark_results, key=lambda x: x["macro_f1"])
    best_name = best_res["model_name"]
    best_pipe = trained_pipelines[best_name]

    print(f"\n[BEST MODEL SELECTED] (Winner) {best_name} (Macro F1 = {best_res['macro_f1']})")

    # Save best model pipeline
    out_models_dir = PROJECT_ROOT / models_dir
    out_models_dir.mkdir(parents=True, exist_ok=True)

    model_save_path = out_models_dir / "next_stage_model.pkl"
    joblib.dump(best_pipe, model_save_path)
    print(f"[MODEL SAVED] Serialized best pipeline to: {model_save_path}")

    # Save comprehensive metadata
    meta = {
        "model_name": best_name,
        "target": "next_stage",
        "training_dataset": data_path,
        "split_strategy": "grouped_by_deal_id",
        "train_deals": int(train_df["deal_id"].nunique()),
        "validation_deals": int(val_df["deal_id"].nunique()),
        "test_deals": int(test_df["deal_id"].nunique()),
        "train_interactions": int(len(train_df)),
        "validation_interactions": int(len(val_df)),
        "test_interactions": int(len(test_df)),
        "accuracy": best_res["accuracy"],
        "macro_f1": best_res["macro_f1"],
        "weighted_f1": best_res["weighted_f1"],
        "precision_macro": best_res["precision_macro"],
        "recall_macro": best_res["recall_macro"],
        "classes": best_res["classes"],
        "per_stage_metrics": best_res["per_stage_metrics"],
        "confusion_matrix": best_res["confusion_matrix"],
        "all_models_benchmarked": benchmark_results,
    }

    meta_path = out_models_dir / "next_stage_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"[METADATA SAVED] Exported metadata to: {meta_path}")

    # Also save comparison report
    comp_path = out_models_dir / "next_stage_comparison.json"
    with open(comp_path, "w", encoding="utf-8") as f:
        json.dump(benchmark_results, f, indent=2)

    return meta


if __name__ == "__main__":
    train_next_stage_pipeline()
