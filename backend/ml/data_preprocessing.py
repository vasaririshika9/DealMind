"""
DealSight AI - Data Preprocessing & Feature Engineering Pipeline
Handles data cleaning, leakage prevention, engineered domain features,
and scikit-learn ColumnTransformer preprocessing for model training & inference.
"""

import os
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Any
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

# Define base numerical features (strictly pre-outcome)
BASE_NUMERIC_FEATURES = [
    "deal_value",
    "total_calls",
    "days_since_last_call",
    "calls_last_7_days",
    "calls_last_30_days",
    "price_objections",
    "competitor_mentions",
    "demo_requested",
    "demo_completed",
    "decision_maker_present",
    "decision_maker_engagement",
    "followups",
    "followup_response_rate",
    "response_delay_hours",
    "sentiment_score",
    "sentiment_trend",
    "engagement_score",
    "objection_count",
    "objection_resolution_rate",
    "budget_confirmed",
    "timeline_confirmed",
    "competitor_present",
    "discount_requested",
    "proposal_sent",
    "proposal_age_days",
    "stage_duration_days",
    "customer_interest_score",
    "next_action_completed",
]

# Engineered numerical features derived from interactions
ENGINEERED_NUMERIC_FEATURES = [
    "engagement_velocity",
    "objection_pressure",
    "decision_maker_score",
    "response_health",
    "stall_risk",
    "unresolved_objection_ratio",
    "pricing_risk",
    "competitor_risk",
    "deal_momentum",
    "interaction_frequency",
]

# Categorical features
CATEGORICAL_FEATURES = [
    "industry",
    "company_size",
    "deal_stage",
    "previous_stage",
]

# All predictor columns
NUMERIC_FEATURES = BASE_NUMERIC_FEATURES + ENGINEERED_NUMERIC_FEATURES
ALL_PREDICTOR_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derives domain-specific sales intelligence features without target leakage."""
    df = df.copy()

    # Engagement velocity: ratio of recent call cadence to monthly call cadence
    calls_30 = df["calls_last_30_days"].clip(lower=0)
    calls_7 = df["calls_last_7_days"].clip(lower=0)
    df["engagement_velocity"] = calls_7 / (calls_30 + 1.0)

    # Objection pressure: weighted penalty for pricing objections and competitor threats
    df["objection_pressure"] = (
        df["price_objections"] * 1.5 + df["competitor_mentions"] * 1.2
    )

    # Decision maker involvement composite
    dm_present = df["decision_maker_present"].fillna(0)
    dm_engage = df["decision_maker_engagement"].fillna(0)
    df["decision_maker_score"] = dm_present * (1.0 + dm_engage)

    # Response health: fast response & high reply rate indicates high intent
    delay_hours = df["response_delay_hours"].clip(lower=0)
    reply_rate = df["followup_response_rate"].clip(lower=0, upper=1)
    df["response_health"] = reply_rate / (1.0 + np.log1p(delay_hours))

    # Stall risk: deals with long inactivity relative to stage duration
    stage_days = df["stage_duration_days"].clip(lower=1)
    days_inactive = df["days_since_last_call"].clip(lower=0)
    df["stall_risk"] = days_inactive / stage_days

    # Unresolved objections ratio
    res_rate = df["objection_resolution_rate"].fillna(1.0).clip(0, 1)
    total_obj = df["objection_count"].clip(lower=0)
    df["unresolved_objection_ratio"] = (1.0 - res_rate) * total_obj

    # Pricing risk: pricing objections amplified when discount is already requested
    disc_req = df["discount_requested"].fillna(0)
    df["pricing_risk"] = df["price_objections"] * (1.0 + 0.5 * disc_req)

    # Competitor risk: competitor presence combined with frequency of mentions
    comp_present = df["competitor_present"].fillna(0)
    df["competitor_risk"] = df["competitor_mentions"] * (1.0 + comp_present)

    # Deal momentum: positive sentiment trend amplified by engagement
    sent_score = df["sentiment_score"].fillna(0.5)
    sent_trend = df["sentiment_trend"].fillna(0)
    eng_score = df["engagement_score"].fillna(0.5)
    df["deal_momentum"] = sent_score * (1.0 + 0.3 * sent_trend) * eng_score

    # Interaction frequency: total calls normalized by stage duration
    df["interaction_frequency"] = df["total_calls"] / stage_days

    return df


def build_preprocessor() -> ColumnTransformer:
    """Builds a scikit-learn ColumnTransformer with Imputation, Scaling & One-Hot Encoding."""
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ]
    )

    return preprocessor


def load_and_preprocess_data(
    file_path: str = "data/sales_data.csv",
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, ColumnTransformer]:
    """Loads sales data, performs cleaning, feature engineering, and stratified 70/15/15 split.
    
    Returns:
        (X_train, X_val, X_test, y_train, y_val, y_test, fitted_preprocessor)
    """
    if not os.path.exists(file_path):
        backend_candidate = Path(__file__).resolve().parent.parent / "data" / "sales_data.csv"
        root_candidate = Path(__file__).resolve().parent.parent.parent / "data" / "sales_data.csv"
        if backend_candidate.exists():
            file_path = str(backend_candidate)
        elif root_candidate.exists():
            file_path = str(root_candidate)
        else:
            raise FileNotFoundError(f"Sales dataset not found at {file_path}")

    raw_df = pd.read_csv(file_path)

    # Clean duplicates
    raw_df = raw_df.drop_duplicates(subset=["deal_id"])

    # Ensure required target column exists
    if "deal_outcome" not in raw_df.columns:
        raise ValueError("Missing 'deal_outcome' target column in dataset.")

    # Apply Feature Engineering
    df = engineer_features(raw_df)

    # Separate target (1 = Progressed, 0 = Lost)
    y = (df["deal_outcome"].str.strip() == "Progressed").astype(int)

    # Drop identifiers and targets from X to prevent data leakage
    drop_cols = ["deal_id", "customer", "deal_outcome"]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # First split: 70% train, 30% temp (which becomes 15% val + 15% test)
    temp_ratio = val_size + test_size
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=temp_ratio, stratify=y, random_state=random_state
    )

    # Second split: 50% / 50% of the temp set -> 15% val and 15% test of total
    val_ratio_of_temp = val_size / temp_ratio
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=(1.0 - val_ratio_of_temp), stratify=y_temp, random_state=random_state
    )

    # Fit preprocessor on training data ONLY to prevent data leakage
    preprocessor = build_preprocessor()
    preprocessor.fit(X_train)

    print(f"[DATA PREPROCESSING] Total samples: {len(df)}")
    print(f"[DATA PREPROCESSING] Train: {len(X_train)} ({(len(X_train)/len(df)):.1%}) | "
          f"Val: {len(X_val)} ({(len(X_val)/len(df)):.1%}) | "
          f"Test: {len(X_test)} ({(len(X_test)/len(df)):.1%})")
    print(f"[DATA PREPROCESSING] Total features: {len(ALL_PREDICTOR_FEATURES)} "
          f"({len(NUMERIC_FEATURES)} numeric, {len(CATEGORICAL_FEATURES)} categorical)")

    return X_train, X_val, X_test, y_train, y_val, y_test, preprocessor


if __name__ == "__main__":
    X_tr, X_v, X_te, y_tr, y_v, y_te, prep = load_and_preprocess_data()
    print("Preprocessing completed successfully.")
