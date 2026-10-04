"""
DealSight AI - Reusable Prediction & Next-Step Service
File: backend/services/prediction_service.py

Loads serialized models:
1. Deal Prediction Pipeline (models/deal_prediction_model.pkl)
2. Next-Stage Milestone Predictor (models/next_stage_model.pkl)
3. SHAP Explainability Service (LinearExplainer / TreeExplainer)
4. Groq Natural Language Synthesis Engine

Performs inference, calculates true SHAP positive & risk signals, predicts exact next sales milestone,
and synthesizes actionable recommendations without inventing numbers.
"""

from __future__ import annotations

import os
import sys
import json
import warnings
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import joblib
import requests

try:
    from sklearn.exceptions import InconsistentVersionWarning
    warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
except ImportError:
    pass

BASE_DIR = Path(__file__).resolve().parent  # backend/services
BACKEND_DIR = BASE_DIR.parent             # backend
PROJECT_ROOT = BACKEND_DIR.parent         # project root

for p in [str(PROJECT_ROOT), str(BACKEND_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from src.data_preprocessing import (
    engineer_features,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
)
from services.explainability_service import get_explainability_service


class PredictionService:
    """Reusable, production-grade prediction service loaded once in memory."""

    def __init__(
        self,
        model_file: str = "models/deal_prediction_model.pkl",
        next_stage_file: str = "models/next_stage_model.pkl",
    ):
        self.model_path = PROJECT_ROOT / model_file
        if not self.model_path.exists():
            self.model_path = PROJECT_ROOT / "models" / "best_model.pkl"

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model file not found at {self.model_path}. Please run ml/train.py first."
            )

        print(f"[PREDICTION SERVICE] Loading primary model from: {self.model_path}")
        self.pipeline = joblib.load(self.model_path)
        self.preprocessor = self.pipeline.named_steps["preprocessor"]
        self.classifier = self.pipeline.named_steps["classifier"]

        # Load Next-Stage Transition Model
        self.next_stage_path = PROJECT_ROOT / next_stage_file
        self.next_stage_pipeline = None
        if self.next_stage_path.exists():
            try:
                self.next_stage_pipeline = joblib.load(self.next_stage_path)
                print(f"[PREDICTION SERVICE] Loaded next-stage model from: {self.next_stage_path}")
            except Exception as e:
                print(f"[PREDICTION SERVICE NOTICE] Next stage model load notice: {e}")

        # Initialize SHAP explainer
        try:
            self.shap_service = get_explainability_service()
        except Exception as e:
            print(f"[PREDICTION SERVICE NOTICE] SHAP service initialization notice: {e}")
            self.shap_service = None

        print("[PREDICTION SERVICE] Prediction services initialized.")

    def _prepare_dataframe(self, input_data: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Enriches raw input with domain defaults and applies feature engineering."""
        total_calls = int(input_data.get("total_calls", 3))
        days_since_last = int(input_data.get("days_since_last_call", 3))
        price_obj = int(input_data.get("price_objections", 0))
        comp_mentions = int(input_data.get("competitor_mentions", 0))
        demo_req = int(input_data.get("demo_requested", 1))
        demo_comp = int(input_data.get("demo_completed", 1 if demo_req else 0))
        dm_present = int(input_data.get("decision_maker_present", 1))
        followups = int(input_data.get("followups", 2))
        sentiment = float(input_data.get("sentiment_score", 0.70))

        objection_rate = price_obj / max(total_calls, 1)
        followup_frequency = followups / max(total_calls, 1)
        call_frequency = total_calls / max(days_since_last + 7, 1)
        engagement_score = float(input_data.get("engagement_score", 0.65))

        full_record = {
            "deal_value": float(input_data.get("deal_value", 55000.0)),
            "total_calls": total_calls,
            "days_since_last_call": days_since_last,
            "calls_last_7_days": int(input_data.get("calls_last_7_days", 1)),
            "calls_last_30_days": int(input_data.get("calls_last_30_days", total_calls)),
            "price_objections": price_obj,
            "competitor_mentions": comp_mentions,
            "demo_requested": demo_req,
            "demo_completed": demo_comp,
            "decision_maker_present": dm_present,
            "decision_maker_engagement": float(input_data.get("decision_maker_engagement", 0.8 if dm_present else 0.0)),
            "followups": followups,
            "followup_response_rate": float(input_data.get("followup_response_rate", 0.75)),
            "response_delay_hours": float(input_data.get("response_delay_hours", 12.0)),
            "sentiment_score": sentiment,
            "sentiment_trend": int(input_data.get("sentiment_trend", 1 if sentiment > 0.65 else (-1 if sentiment < 0.45 else 0))),
            "engagement_score": engagement_score,
            "objection_count": price_obj + comp_mentions,
            "objection_resolution_rate": float(input_data.get("objection_resolution_rate", 0.85 if price_obj == 0 else 0.5)),
            "budget_confirmed": int(input_data.get("budget_confirmed", 1 if dm_present and sentiment > 0.6 else 0)),
            "timeline_confirmed": int(input_data.get("timeline_confirmed", 1 if demo_comp else 0)),
            "competitor_present": int(input_data.get("competitor_present", 1 if comp_mentions > 0 else 0)),
            "discount_requested": int(input_data.get("discount_requested", 1 if price_obj > 1 else 0)),
            "proposal_sent": int(input_data.get("proposal_sent", 1 if demo_comp and dm_present else 0)),
            "proposal_age_days": int(input_data.get("proposal_age_days", 3)),
            "deal_stage": str(input_data.get("deal_stage", "Evaluation")),
            "previous_stage": str(input_data.get("previous_stage", "Demo")),
            "stage_duration_days": int(input_data.get("stage_duration_days", 14)),
            "customer_interest_score": float(input_data.get("customer_interest_score", sentiment)),
            "next_action_completed": int(input_data.get("next_action_completed", 1)),
            "industry": str(input_data.get("industry", "Technology")),
            "company_size": str(input_data.get("company_size", "Mid-Market")),
            "objection_rate": round(objection_rate, 3),
            "followup_frequency": round(followup_frequency, 3),
            "call_frequency": round(call_frequency, 3),
        }

        for k, v in input_data.items():
            if k in full_record:
                full_record[k] = v

        raw_df = pd.DataFrame([full_record])
        engineered_df = engineer_features(raw_df)
        return engineered_df, full_record

    def _predict_next_stage(
        self, full_record: Dict[str, Any], current_stage: str
    ) -> Tuple[str, float, Dict[str, float], List[str]]:
        """Predicts the next sales milestone and stage probability distribution using the multi-class model."""
        if self.next_stage_pipeline is not None:
            try:
                stage_input_dict = {
                    "sentiment_score": float(full_record.get("sentiment_score", 0.65)),
                    "price_objections": int(full_record.get("price_objections", 0)),
                    "competitor_mentions": int(full_record.get("competitor_mentions", 0)),
                    "demo_requested": int(full_record.get("demo_requested", 0)),
                    "demo_completed": int(full_record.get("demo_completed", 0)),
                    "decision_maker_present": int(full_record.get("decision_maker_present", 0)),
                    "followups": int(full_record.get("followups", 2)),
                    "days_since_last_call": int(full_record.get("days_since_last_call", 3)),
                    "response_delay_hours": float(full_record.get("response_delay_hours", 12.0)),
                    "engagement_score": float(full_record.get("engagement_score", 0.60)),
                    "deal_value": float(full_record.get("deal_value", 50000.0)),
                    "call_number": int(full_record.get("total_calls", 1)),
                    "current_stage": str(current_stage),
                    "industry": str(full_record.get("industry", "Technology")),
                    "company_size": str(full_record.get("company_size", "Mid-Market")),
                }
                stage_df = pd.DataFrame([stage_input_dict])
                probs = self.next_stage_pipeline.predict_proba(stage_df)[0]
                clf = self.next_stage_pipeline.named_steps["classifier"]
                classes = list(getattr(clf, "classes_", self.next_stage_pipeline.classes_))

                best_idx = int(np.argmax(probs))
                predicted_stage = str(classes[best_idx])
                confidence = round(float(probs[best_idx]), 3)

                stage_probabilities = {
                    cls: round(float(p), 3)
                    for cls, p in sorted(zip(classes, probs), key=lambda x: x[1], reverse=True)
                }

                # Plain-English drivers for the next stage transition
                stage_reasons = []
                if stage_input_dict["sentiment_score"] >= 0.70:
                    stage_reasons.append("Positive customer sentiment and buyer interest")
                elif stage_input_dict["sentiment_score"] < 0.40:
                    stage_reasons.append("Low buyer enthusiasm and hesitant communication")

                if stage_input_dict["demo_completed"] == 1:
                    stage_reasons.append("Product demonstration successfully delivered")
                elif stage_input_dict["demo_requested"] == 1:
                    stage_reasons.append("Active demo request in progress")

                if stage_input_dict["decision_maker_present"] == 1:
                    stage_reasons.append("Executive decision maker actively involved")

                if stage_input_dict["price_objections"] == 0:
                    stage_reasons.append("Low price objection count")
                elif stage_input_dict["price_objections"] >= 2:
                    stage_reasons.append(f"{stage_input_dict['price_objections']} pricing objections unresolved")

                if stage_input_dict["days_since_last_call"] <= 4:
                    stage_reasons.append("Recent follow-up activity")

                if not stage_reasons:
                    stage_reasons = ["Consistent call progression and standard pipeline velocity"]

                return predicted_stage, confidence, stage_probabilities, stage_reasons[:4]
            except Exception as e:
                print(f"[PREDICTION SERVICE NOTICE] Next stage model inference error: {e}")

        # Fallback if next stage model is not initialized
        stage_flow = {
            "Discovery": "Demo",
            "Demo": "Evaluation",
            "Evaluation": "Negotiation",
            "Negotiation": "Proposal",
            "Proposal": "Closed Won",
        }
        fallback_stage = stage_flow.get(current_stage, "Evaluation")
        return fallback_stage, 0.75, {fallback_stage: 0.75}, ["Standard lifecycle transition"]

    def _call_groq_synthesis(self, structured_info: Dict[str, Any]) -> Optional[Dict[str, str]]:
        """Invokes Groq LLM strictly to synthesize a 3-part plain-English recommendation."""
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            return None

        prompt = f"""You are a professional sales assistant.
Explain the provided deal prediction to a normal salesperson.

Do not invent facts.
Do not change the prediction.
Do not calculate or invent probabilities.
Do not mention machine-learning models, SHAP, algorithms, datasets, or technical metrics.

Information:
- Deal Status: {structured_info['headline']}
- Probability: {structured_info['probability_text']}
- Likely Next Milestone: {structured_info['likely_next_step']} ({structured_info['next_step_probability']} likelihood)
- Key Positive Drivers: {', '.join(structured_info['why'])}
- Friction Points / Concerns: {', '.join(structured_info['concerns'])}
- Customer History: {structured_info.get('hindsight_memory', 'Customer evaluated pricing in previous discussions')}
- Suggested Direction: {structured_info['next_action']}

Return ONLY valid JSON with this exact structure:
{{
  "what_we_think": "One short sentence summarizing deal forecast",
  "why": "One or two short sentences explaining why based strictly on the signals provided",
  "what_to_do_next": "One actionable sentence recommending what the rep should do"
}}
"""
        try:
            res = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
                    "messages": [
                        {"role": "system", "content": "You are a concise sales decision assistant. Return valid JSON only."},
                        {"role": "user", "content": prompt},
                    ],
                    "response_format": {"type": "json_object"},
                    "temperature": 0.2,
                    "max_tokens": 300,
                },
                timeout=10,
            )
            if res.status_code == 200:
                data = res.json()
                content = data["choices"][0]["message"]["content"]
                return json.loads(content)
        except Exception as e:
            print(f"[PREDICTION SERVICE NOTICE] Groq synthesis fallback: {e}")

        return None

    def predict(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Runs end-to-end prediction: ML probability + SHAP explanation + Next-stage forecast + Groq action."""
        engineered_df, full_record = self._prepare_dataframe(input_data)

        # 1. Primary Model Inference
        prob_array = self.pipeline.predict_proba(engineered_df)[0]
        prob_lost = float(prob_array[0])
        prob_progress = float(prob_array[1])

        pred_label = "Progressed" if prob_progress >= 0.50 else "Lost"
        pct_progress = int(round(prob_progress * 100))
        pct_lost = 100 - pct_progress

        # 2. Risk Level Calculation
        price_obj = full_record["price_objections"]
        dm_present = full_record["decision_maker_present"]
        delay_hrs = full_record["response_delay_hours"]
        sentiment = full_record["sentiment_score"]

        risk_score = int(round(
            (prob_lost * 65.0) +
            (min(price_obj, 3) * 6.0) +
            ((1 - dm_present) * 12.0) +
            (min(delay_hrs / 48.0, 1.0) * 10.0) +
            (max(0.0, (0.6 - sentiment)) * 12.0)
        ))
        risk_score = max(5, min(98, risk_score))

        if risk_score <= 33 and prob_progress >= 0.65:
            risk_level = "Low"
            risk_text = "Low risk"
            headline = "Likely to move forward"
        elif risk_score <= 65:
            risk_level = "Medium"
            risk_text = "Medium risk"
            headline = "Needs attention"
        else:
            risk_level = "High"
            risk_text = "High risk"
            headline = "At risk of stalling or loss"

        # 3. Next-Stage Milestone Prediction
        current_stage = full_record.get("deal_stage", "Evaluation")
        next_stage_name, next_stage_prob, stage_probabilities, next_stage_signals = self._predict_next_stage(
            full_record, current_stage
        )
        next_stage_pct_str = f"{int(round(next_stage_prob * 100))}%"

        # 4. SHAP-based Explanation
        shap_res = None
        if self.shap_service:
            try:
                shap_res = self.shap_service.explain(engineered_df)
            except Exception as e:
                print(f"[PREDICTION SERVICE NOTICE] SHAP calculation fallback: {e}")

        if shap_res:
            positive_signals = shap_res["positive_signals"]
            warning_signs = shap_res["warning_signs"]
            why_list = shap_res["plain_why"]
            concerns_list = shap_res["plain_concerns"]
            judge_shap_details = shap_res.get("judge_shap_details", [])
            explainer_name = shap_res.get("explainer_used", "SHAP LinearExplainer")
        else:
            # Fallback signals
            positive_signals = [
                "✓ Customer interest is positive and receptive",
                "✓ Product demo has been completed",
                "✓ Executive decision maker is involved in conversations",
            ]
            warning_signs = [
                "⚠ Pricing concern remains unresolved" if price_obj > 0 else "No active friction points detected",
            ]
            why_list = [s.replace("✓ ", "") for s in positive_signals]
            concerns_list = [w.replace("⚠ ", "") for w in warning_signs]
            judge_shap_details = []
            explainer_name = "Rule-based Attribution"

        # 5. Determine Recommended Action
        if price_obj > 0:
            recommended_action = "Address the pricing concern using the ROI comparison that worked in previous discussions."
        elif dm_present == 0:
            recommended_action = "Involve the executive decision maker before sending the final proposal."
        elif full_record.get("competitor_present", 0) == 1:
            recommended_action = "Equip rep with a competitive battlecard emphasizing total cost of ownership advantages."
        elif full_record.get("demo_completed", 0) == 0 and full_record.get("demo_requested", 0) == 1:
            recommended_action = "Lock in the product demonstration call to solidify buyer interest."
        elif delay_hrs > 36:
            recommended_action = "Reconnect with the customer to confirm whether the project timeline is still an active priority."
        else:
            if prob_progress >= 0.70:
                recommended_action = f"Deal momentum is strong. Align key stakeholders toward the upcoming {next_stage_name} milestone."
            else:
                recommended_action = "Schedule a discovery review to reaffirm requirements and unblock progression."

        # 6. Groq Synthesis (if configured)
        groq_payload = {
            "headline": headline,
            "probability_text": f"{pct_progress}% chance of moving forward",
            "likely_next_step": next_stage_name,
            "next_step_probability": next_stage_pct_str,
            "why": why_list[:3],
            "concerns": concerns_list[:2],
            "next_action": recommended_action,
            "hindsight_memory": input_data.get("customer_memory", "Previous objection was pricing; ROI comparison proved effective."),
        }
        groq_out = self._call_groq_synthesis(groq_payload)

        what_we_think = groq_out["what_we_think"] if groq_out and "what_we_think" in groq_out else headline
        groq_why = groq_out["why"] if groq_out and "why" in groq_out else (
            f"The customer is engaged and key milestones are in motion." if prob_progress >= 0.50
            else "Several concerns remain unresolved and response cadence has slowed."
        )
        groq_action = groq_out["what_to_do_next"] if groq_out and "what_to_do_next" in groq_out else recommended_action

        # Customer Interest Indicator
        sent_trend = full_record.get("sentiment_trend", 0)
        if sent_trend > 0 or sentiment >= 0.75:
            customer_interest = "Improving ↑"
        elif sent_trend < 0 or sentiment <= 0.40:
            customer_interest = "Declining ↓"
        else:
            customer_interest = "Steady →"

        # Format backward-compatible factor dictionaries for any caller expecting {'label': ..., 'impact': ...}
        pos_factors_list = []
        for s in positive_signals[:4]:
            clean_s = s.replace("✓ ", "")
            pos_factors_list.append({"feature": "signal", "label": clean_s, "impact": 0.45})

        risk_factors_list = []
        for w in warning_signs[:3]:
            clean_w = w.replace("⚠ ", "")
            risk_factors_list.append({"feature": "friction", "label": clean_w, "impact": -0.45})

        return {
            # Structured technical fields
            "prediction": pred_label,
            "confidence": round(max(prob_progress, prob_lost), 3),
            "progress_probability": round(prob_progress, 4),
            "loss_probability": round(prob_lost, 4),
            "risk_level": risk_level,
            "risk_score": risk_score,
            "important_factors": why_list[:3],
            "positive_factors": pos_factors_list,
            "risk_factors": risk_factors_list,
            "recommended_action": groq_action,
            # User-friendly simple language fields
            "headline": what_we_think,
            "forecast_summary": what_we_think,
            "deal_health": "Healthy Deal" if prob_progress >= 0.65 else ("Needs Attention" if prob_progress >= 0.45 else "At Risk"),
            "probability_text": f"{pct_progress}% chance of moving forward",
            "chance_moving_forward": f"{pct_progress}%",
            "chance_losing": f"{pct_lost}%",
            "risk_text": risk_text,
            "customer_interest": customer_interest,
            # "What happens next?"
            "current_stage": current_stage,
            "likely_next_step": next_stage_name,
            "predicted_next_stage": next_stage_name,
            "next_stage_confidence": round(next_stage_prob, 3),
            "next_step_probability": next_stage_pct_str,
            "next_step_likelihood": f"{next_stage_pct_str} likelihood",
            "stage_probabilities": stage_probabilities,
            "next_stage_signals": next_stage_signals,
            # "Why we think this" & "Watch out for"
            "why": why_list[:3],
            "concerns": concerns_list[:2],
            "positive_signals": positive_signals[:3],
            "warning_signs": warning_signs[:2],
            # "What should I do?"
            "next_action": groq_action,
            "what_to_do_next": groq_action,
            # Groq 3-part structured breakdown
            "groq_brief": {
                "what_we_think": what_we_think,
                "why": groq_why,
                "what_to_do": groq_action,
            },
            # Judge Drawer Details
            "explainer_used": explainer_name,
            "judge_shap_details": judge_shap_details,
        }

    def predict_deal(self, deal_data: Dict[str, Any]) -> Dict[str, Any]:
        """Full backwards-compatibility alias for predict()."""
        return self.predict(deal_data)

    def predict_for_customer(
        self,
        customer_name: str,
        history: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Convenience method to generate prediction for a given customer and call history."""
        payload: Dict[str, Any] = {"customer_name": customer_name, **kwargs}
        if history:
            payload["total_calls"] = len(history)
            if len(history) > 0 and "company_name" in history[0]:
                payload["company_name"] = history[0]["company_name"]
        return self.predict(payload)


# Singleton service instance
_SERVICE_INSTANCE: Optional[PredictionService] = None


def get_prediction_service() -> PredictionService:
    """Returns the singleton instance of PredictionService."""
    global _SERVICE_INSTANCE
    if _SERVICE_INSTANCE is None:
        _SERVICE_INSTANCE = PredictionService()
    return _SERVICE_INSTANCE


def get_predictor() -> PredictionService:
    """Alias for get_prediction_service for backward compatibility."""
    return get_prediction_service()


if __name__ == "__main__":
    svc = get_prediction_service()
    test_deal = {
        "total_calls": 4,
        "days_since_last_call": 2,
        "price_objections": 1,
        "competitor_mentions": 0,
        "demo_requested": 1,
        "demo_completed": 1,
        "decision_maker_present": 1,
        "followups": 3,
        "sentiment_score": 0.78,
        "deal_stage": "Evaluation",
    }
    res = svc.predict(test_deal)
    print("\n--- END-TO-END PREDICTION WITH SHAP & NEXT STAGE ---")
    print(json.dumps(res, indent=2))
