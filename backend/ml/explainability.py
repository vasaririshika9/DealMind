"""
DealSight AI - Explainable AI & Next-Best-Action Engine
Computes localized feature contributions (Linear Shapley / Tree importances),
calculates business risk level (Low / Medium / High), predicts next stage,
and generates contextual sales recommendations tied to detected risk factors.
"""

import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
from pathlib import Path
import sys

BACKEND_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = BACKEND_DIR.parent
ML_DIR = Path(__file__).resolve().parent

for p in [str(BACKEND_DIR), str(PROJECT_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from ml.data_preprocessing import (
        engineer_features,
        NUMERIC_FEATURES,
        CATEGORICAL_FEATURES,
    )
except ImportError:
    from backend.ml.data_preprocessing import (
        engineer_features,
        NUMERIC_FEATURES,
        CATEGORICAL_FEATURES,
    )

STAGES_ORDER = ["Discovery", "Demo", "Evaluation", "Negotiation", "Proposal", "Closed Won"]

# Human-friendly descriptions for key features
FEATURE_LABELS = {
    "price_objections": "Price Objections",
    "objection_pressure": "Objection Pressure",
    "decision_maker_present": "Decision Maker Present",
    "decision_maker_engagement": "Decision Maker Engagement",
    "demo_completed": "Demo Completed",
    "budget_confirmed": "Budget Confirmed",
    "timeline_confirmed": "Timeline Confirmed",
    "response_delay_hours": "Customer Response Delay",
    "days_since_last_call": "Days Inactive",
    "sentiment_score": "Call Sentiment",
    "sentiment_trend": "Sentiment Trend",
    "engagement_score": "Customer Engagement",
    "competitor_present": "Competitor Presence",
    "competitor_mentions": "Competitor Mentions",
    "proposal_sent": "Proposal Sent",
    "proposal_age_days": "Proposal Age",
    "stage_duration_days": "Days in Current Stage",
    "objection_resolution_rate": "Objection Resolution Rate",
    "next_action_completed": "Next Action Follow-Through",
    "followup_response_rate": "Follow-Up Response Rate",
    "stall_risk": "Deal Inactivity / Stall Risk",
    "deal_momentum": "Deal Momentum Composite",
    "response_health": "Response Health Ratio",
    "customer_interest_score": "Customer Interest Score",
}


class DealPredictor:
    """Production predictor with Explainable AI and Risk Scoring."""

    def __init__(self, model_path: str = None):
        if not model_path:
            candidates = [
                BACKEND_DIR / "models" / "deal_prediction_model.pkl",
                BACKEND_DIR / "models" / "best_model.pkl",
                PROJECT_ROOT / "models" / "deal_prediction_model.pkl",
                PROJECT_ROOT / "models" / "best_model.pkl",
            ]
            found = next((p for p in candidates if p.exists()), None)
            model_path = str(found) if found else str(BACKEND_DIR / "models" / "deal_prediction_model.pkl")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}. Train model first.")
        self.pipeline = joblib.load(model_path)
        self.preprocessor = self.pipeline.named_steps["preprocessor"]
        self.classifier = self.pipeline.named_steps["classifier"]

    def _extract_feature_contributions(self, single_row_df: pd.DataFrame) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Calculates localized feature contributions for the prediction."""
        X_trans = self.preprocessor.transform(single_row_df)

        feature_names = list(NUMERIC_FEATURES)
        try:
            cat_encoder = self.preprocessor.named_transformers_["cat"].named_steps["onehot"]
            cat_names = cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES)
            feature_names.extend(cat_names.tolist())
        except Exception:
            pass

        contributions = []
        if hasattr(self.classifier, "coef_"):
            # Logistic Regression: contribution_i = coef_i * x_transformed_i
            coefs = self.classifier.coef_[0]
            row_vals = X_trans[0]
            for name, val, coef in zip(feature_names, row_vals, coefs):
                impact = float(val * coef)
                label = FEATURE_LABELS.get(name, name.replace("_", " ").title())
                contributions.append({
                    "feature": name,
                    "label": label,
                    "impact": round(impact, 3),
                })
        else:
            # Tree-based fallback
            importances = getattr(self.classifier, "feature_importances_", np.ones(len(feature_names)))
            for name, imp in zip(feature_names, importances):
                label = FEATURE_LABELS.get(name, name.replace("_", " ").title())
                contributions.append({
                    "feature": name,
                    "label": label,
                    "impact": round(float(imp), 3),
                })

        # Separate positive (pushing towards Progressed) and negative (pushing towards Lost)
        positive_factors = sorted([c for c in contributions if c["impact"] > 0], key=lambda x: x["impact"], reverse=True)[:5]
        risk_factors = sorted([c for c in contributions if c["impact"] < 0], key=lambda x: x["impact"])[:5]

        return positive_factors, risk_factors

    def predict_deal(self, deal_data: Dict[str, Any]) -> Dict[str, Any]:
        """Predicts deal progression probability, risk score, factors, and recommended action."""
        # Convert dictionary to DataFrame
        row_df = pd.DataFrame([deal_data])

        # Fill default baseline fields if missing
        defaults = {
            "industry": "Technology",
            "company_size": "Mid-Market",
            "deal_value": 50000.0,
            "total_calls": 3,
            "days_since_last_call": 3,
            "calls_last_7_days": 1,
            "calls_last_30_days": 3,
            "price_objections": 0,
            "competitor_mentions": 0,
            "demo_requested": 1,
            "demo_completed": 0,
            "decision_maker_present": 0,
            "decision_maker_engagement": 0.0,
            "followups": 2,
            "followup_response_rate": 0.7,
            "response_delay_hours": 12.0,
            "sentiment_score": 0.65,
            "sentiment_trend": 0,
            "engagement_score": 0.60,
            "objection_count": 0,
            "objection_resolution_rate": 1.0,
            "budget_confirmed": 0,
            "timeline_confirmed": 0,
            "competitor_present": 0,
            "discount_requested": 0,
            "proposal_sent": 0,
            "proposal_age_days": 0,
            "deal_stage": "Evaluation",
            "previous_stage": "Demo",
            "stage_duration_days": 14,
            "customer_interest_score": 0.70,
            "next_action_completed": 1,
        }
        for k, v in defaults.items():
            if k not in row_df.columns:
                row_df[k] = v

        # Apply feature engineering
        engineered_df = engineer_features(row_df)

        # Generate probabilities
        probs = self.pipeline.predict_proba(engineered_df)[0]
        # class 0 = Lost, class 1 = Progressed
        prob_lost = float(probs[0])
        prob_progress = float(probs[1])

        prediction = "Progressed" if prob_progress >= 0.5 else "Lost"
        confidence = round(max(prob_progress, prob_lost), 3)

        # Compute Explainable AI factors
        positive_factors, risk_factors = self._extract_feature_contributions(engineered_df)

        # Risk Score Calculation (0 to 100)
        # Scaled primarily by probability of loss + penalty for high unaddressed risks
        base_risk_score = prob_lost * 100.0
        risk_adjustment = 0.0

        if deal_data.get("price_objections", 0) >= 2:
            risk_adjustment += 6.0
        if deal_data.get("decision_maker_present", 0) == 0:
            risk_adjustment += 8.0
        if deal_data.get("competitor_present", 0) == 1:
            risk_adjustment += 5.0
        if deal_data.get("days_since_last_call", 0) > 14:
            risk_adjustment += 7.0

        risk_score = int(np.clip(base_risk_score + risk_adjustment, 0, 100))

        if risk_score <= 30:
            risk_level = "Low"
        elif risk_score <= 60:
            risk_level = "Medium"
        else:
            risk_level = "High"

        # Plain-English Deal Health
        if risk_score <= 30 and prob_progress >= 0.65:
            deal_health = "Healthy Deal"
        elif risk_score <= 60 and prob_progress >= 0.35:
            deal_health = "Needs Attention"
        else:
            deal_health = "At Risk"

        forecast_summary = "Likely to move forward" if prediction == "Progressed" else "Likely to stall or be lost"

        # Customer Interest & Engagement Level
        sent_trend = deal_data.get("sentiment_trend", 0)
        sent_score = deal_data.get("sentiment_score", 0.6)
        if sent_trend > 0 or sent_score >= 0.75:
            customer_interest = "Improving ↑"
        elif sent_trend < 0 or sent_score <= 0.40:
            customer_interest = "Declining ↓"
        else:
            customer_interest = "Steady →"

        eng_score = deal_data.get("engagement_score", 0.6)
        if eng_score >= 0.70:
            customer_interest_level = "Strong"
        elif eng_score >= 0.45:
            customer_interest_level = "Moderate"
        else:
            customer_interest_level = "Low"

        # Human-Friendly Signals Checklist
        positive_signals = []
        if deal_data.get("decision_maker_present", 0) == 1:
            positive_signals.append("✓ Decision maker is involved in conversations")
        if deal_data.get("demo_completed", 0) == 1:
            positive_signals.append("✓ Customer completed the product demonstration")
        if customer_interest == "Improving ↑":
            positive_signals.append("✓ Customer interest is improving across interactions")
        if deal_data.get("response_delay_hours", 24) <= 18:
            positive_signals.append("✓ Customer is responding quickly to communications")
        if deal_data.get("budget_confirmed", 0) == 1:
            positive_signals.append("✓ Budget allocation has been formally confirmed")
        if deal_data.get("timeline_confirmed", 0) == 1:
            positive_signals.append("✓ Implementation timeline is agreed upon")
        if not positive_signals:
            positive_signals.append("✓ Initial discovery interaction established")

        warning_signs = []
        price_obj = deal_data.get("price_objections", 0)
        if price_obj > 0:
            warning_signs.append(f"⚠ {price_obj} pricing concern{'s are' if price_obj > 1 else ' is'} still unresolved")
        if deal_data.get("competitor_present", 0) == 1:
            warning_signs.append("⚠ Competitor is actively being evaluated by client")
        if deal_data.get("decision_maker_present", 0) == 0:
            warning_signs.append("⚠ Executive decision maker has not yet joined calls")
        if deal_data.get("response_delay_hours", 24) > 36:
            warning_signs.append("⚠ Customer response time has slowed down significantly")
        if deal_data.get("days_since_last_call", 0) >= 10:
            warning_signs.append("⚠ Deal has been inactive for over 10 days")
        if not warning_signs:
            warning_signs.append("No active friction points or critical warnings detected")

        # Predicted Next Stage
        current_stage = str(deal_data.get("deal_stage", "Discovery"))
        if prediction == "Progressed":
            try:
                curr_idx = STAGES_ORDER.index(current_stage)
                next_stage = STAGES_ORDER[min(curr_idx + 1, len(STAGES_ORDER) - 1)]
            except ValueError:
                next_stage = "Negotiation"
        else:
            next_stage = "At Risk / On Hold"

        # Next Best Action Generation based on dominant risk factors
        recommended_action = self._determine_next_best_action(deal_data, risk_factors, prob_progress)

        return {
            "prediction": prediction,
            "forecast_summary": forecast_summary,
            "deal_health": deal_health,
            "chance_moving_forward": f"{int(round(prob_progress * 100))}%",
            "chance_losing": f"{int(round(prob_lost * 100))}%",
            "customer_interest": customer_interest,
            "customer_interest_level": customer_interest_level,
            "progress_probability": round(prob_progress, 3),
            "loss_probability": round(prob_lost, 3),
            "risk_score": risk_score,
            "risk_level": risk_level,
            "confidence": confidence,
            "predicted_next_stage": next_stage,
            "likely_next_step": next_stage,
            "positive_signals": positive_signals,
            "warning_signs": warning_signs,
            "positive_factors": positive_factors,
            "risk_factors": risk_factors,
            "recommended_action": recommended_action,
            "what_to_do_next": recommended_action,
        }

    def _determine_next_best_action(
        self,
        deal: Dict[str, Any],
        risk_factors: List[Dict[str, Any]],
        prob_progress: float,
    ) -> str:
        """Determines context-sensitive sales recommendations."""
        price_obj = deal.get("price_objections", 0)
        dm_present = deal.get("decision_maker_present", 0)
        comp_present = deal.get("competitor_present", 0)
        days_inactive = deal.get("days_since_last_call", 0)
        demo_done = deal.get("demo_completed", 0)
        budget_conf = deal.get("budget_confirmed", 0)

        actions = []

        if price_obj > 0:
            actions.append("Present a value-based ROI business case and flexible payment terms to resolve pricing hesitation.")
        if dm_present == 0:
            actions.append("Request an executive multi-threading call to bring the decision maker / budget holder into the loop.")
        if comp_present == 1:
            actions.append("Equip rep with a competitive battlecard emphasizing proprietary workflow and total cost of ownership advantages.")
        if days_inactive >= 10:
            actions.append("Initiate a high-priority re-engagement sequence addressing specific open questions within 48 hours.")
        if demo_done == 0:
            actions.append("Lock in a tailored product demonstration focused on the customer's top two operational pain points.")
        if budget_conf == 0 and prob_progress > 0.6:
            actions.append("Align on formal budget allocation and implementation milestones before issuing proposal.")

        if not actions:
            if prob_progress >= 0.7:
                return "Deal momentum is strong. Accelerate toward proposal signing and mutual close plan with key stakeholders."
            else:
                return "Schedule a structured review meeting to reaffirm requirements and unblock deal progression."

        return " | ".join(actions[:2])


# Global singleton instance for high-speed API responses
_PREDICTOR_INSTANCE = None


def get_predictor() -> DealPredictor:
    """Returns singleton DealPredictor instance."""
    global _PREDICTOR_INSTANCE
    if _PREDICTOR_INSTANCE is None:
        _PREDICTOR_INSTANCE = DealPredictor()
    return _PREDICTOR_INSTANCE


if __name__ == "__main__":
    predictor = get_predictor()
    sample_deal = {
        "deal_value": 75000,
        "total_calls": 4,
        "days_since_last_call": 2,
        "price_objections": 1,
        "competitor_mentions": 0,
        "demo_completed": 1,
        "decision_maker_present": 1,
        "followups": 3,
        "sentiment_score": 0.78,
        "engagement_score": 0.82,
        "budget_confirmed": 1,
        "timeline_confirmed": 1,
        "deal_stage": "Evaluation",
    }
    result = predictor.predict_deal(sample_deal)
    print("Test Prediction Output:")
    import pprint
    pprint.pprint(result)
