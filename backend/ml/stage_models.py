"""
DealSight AI - Next-Stage Model Architecture & Preprocessing
Defines feature schema, ColumnTransformer preprocessor, and model estimators
for multi-class historical stage transition forecasting.
"""

from typing import List
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier

# Numeric features observed strictly at interaction time t
NUMERIC_STAGE_FEATURES: List[str] = [
    "sentiment_score",
    "price_objections",
    "competitor_mentions",
    "demo_requested",
    "demo_completed",
    "decision_maker_present",
    "followups",
    "days_since_last_call",
    "response_delay_hours",
    "engagement_score",
    "deal_value",
    "call_number",
]

# Categorical features known prior to transition
CATEGORICAL_STAGE_FEATURES: List[str] = [
    "current_stage",
    "industry",
    "company_size",
]

ALL_STAGE_FEATURES: List[str] = NUMERIC_STAGE_FEATURES + CATEGORICAL_STAGE_FEATURES


def build_stage_preprocessor() -> ColumnTransformer:
    """Creates a ColumnTransformer to handle imputation, scaling, and one-hot encoding."""
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, NUMERIC_STAGE_FEATURES),
            ("cat", categorical_pipe, CATEGORICAL_STAGE_FEATURES),
        ]
    )


class XGBStringClassifier(BaseEstimator, ClassifierMixin):
    """XGBoost wrapper supporting native string target labels."""

    def __init__(self, **params):
        self.params = params
        self.xgb = XGBClassifier(**params)
        self.classes_ = None

    def fit(self, X, y):
        self.classes_ = np.sort(np.unique(y))
        class_to_idx = {c: i for i, c in enumerate(self.classes_)}
        y_int = np.array([class_to_idx[val] for val in y])
        self.xgb.fit(X, y_int)
        return self

    def predict_proba(self, X):
        return self.xgb.predict_proba(X)

    def predict(self, X):
        probs = self.predict_proba(X)
        return self.classes_[np.argmax(probs, axis=1)]
