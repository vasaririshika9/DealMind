"""
DealSight AI - Deal Feature Extraction Helper
Extracts machine learning feature vectors from customer call histories.
"""

from typing import List, Dict, Any


def extract_features_from_call_history(
    customer_name: str,
    company_name: str,
    history: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Converts a customer's call records into predictive ML feature vector."""
    if not history:
        return {
            "customer": customer_name,
            "company_name": company_name,
            "deal_value": 45000.0,
            "total_calls": 1,
            "days_since_last_call": 3,
            "calls_last_7_days": 1,
            "calls_last_30_days": 1,
            "price_objections": 0,
            "competitor_mentions": 0,
            "demo_requested": 0,
            "demo_completed": 0,
            "decision_maker_present": 0,
            "decision_maker_engagement": 0.0,
            "followups": 1,
            "followup_response_rate": 0.6,
            "response_delay_hours": 24.0,
            "sentiment_score": 0.5,
            "sentiment_trend": 0,
            "engagement_score": 0.5,
            "objection_count": 0,
            "objection_resolution_rate": 1.0,
            "budget_confirmed": 0,
            "timeline_confirmed": 0,
            "competitor_present": 0,
            "discount_requested": 0,
            "proposal_sent": 0,
            "proposal_age_days": 0,
            "deal_stage": "Discovery",
            "previous_stage": "Lead",
            "stage_duration_days": 7,
            "customer_interest_score": 0.5,
            "next_action_completed": 1,
        }

    total_calls = len(history)
    latest_call = history[-1]

    # Calculate price objections
    price_obj_count = 0
    total_obj_count = 0
    resolved_obj_count = 0
    comp_mention_count = 0
    comp_present = 0

    sentiment_map = {
        "positive": 0.85,
        "neutral": 0.55,
        "hesitant": 0.35,
        "negative": 0.18,
    }

    sentiment_history = []

    for c in history:
        objection = str(c.get("objection_raised", "")).lower()
        if objection and objection not in ["none", ""]:
            total_obj_count += 1
            if any(term in objection for term in ["price", "cost", "expensive", "budget", "discount"]):
                price_obj_count += 1

        comp = str(c.get("competitor_mentioned", "")).lower()
        if comp and comp not in ["none", ""]:
            comp_mention_count += 1
            comp_present = 1

        sent_str = str(c.get("sentiment", "neutral")).lower()
        sentiment_history.append(sentiment_map.get(sent_str, 0.55))

    sentiment_score = sentiment_history[-1] if sentiment_history else 0.55
    if len(sentiment_history) > 1:
        diff = sentiment_history[-1] - sentiment_history[0]
        sentiment_trend = 1 if diff > 0.1 else (-1 if diff < -0.1 else 0)
    else:
        sentiment_trend = 0

    latest_stage = latest_call.get("deal_stage", "Discovery")
    prev_stage = history[-2].get("deal_stage", "Lead") if len(history) > 1 else "Lead"

    # Stage inference
    stage_idx_map = {
        "discovery": 0,
        "demo": 1,
        "evaluation": 2,
        "negotiation": 3,
        "proposal": 4,
        "closed won": 5,
    }
    s_idx = stage_idx_map.get(str(latest_stage).lower(), 1)
    demo_completed = 1 if s_idx >= 1 else 0

    # Decision maker presence
    dm_present = 1 if (
        s_idx >= 2 or
        any("decision maker" in str(c).lower() or "vp" in str(c).lower() or "director" in str(c).lower() for c in history)
    ) else 0

    dm_engagement = 0.80 if dm_present and sentiment_score > 0.6 else (0.50 if dm_present else 0.0)

    # Resolution rate
    resolution_rate = 1.0
    if total_obj_count > 0:
        if sentiment_score >= 0.7:
            resolution_rate = 0.85
        elif sentiment_score <= 0.35:
            resolution_rate = 0.25
        else:
            resolution_rate = 0.50

    return {
        "customer": customer_name,
        "company_name": company_name,
        "industry": "Technology",
        "company_size": "Mid-Market",
        "deal_value": 55000.0,
        "total_calls": total_calls,
        "days_since_last_call": 2,
        "calls_last_7_days": min(total_calls, 2),
        "calls_last_30_days": min(total_calls, 4),
        "price_objections": price_obj_count,
        "competitor_mentions": comp_mention_count,
        "demo_requested": 1,
        "demo_completed": demo_completed,
        "decision_maker_present": dm_present,
        "decision_maker_engagement": dm_engagement,
        "followups": min(total_calls + 1, 6),
        "followup_response_rate": 0.75 if sentiment_score >= 0.5 else 0.45,
        "response_delay_hours": 12.0 if sentiment_score >= 0.5 else 36.0,
        "sentiment_score": sentiment_score,
        "sentiment_trend": sentiment_trend,
        "engagement_score": min(0.95, 0.45 + (total_calls * 0.08) + (0.2 * sentiment_score)),
        "objection_count": total_obj_count,
        "objection_resolution_rate": resolution_rate,
        "budget_confirmed": 1 if s_idx >= 2 else 0,
        "timeline_confirmed": 1 if s_idx >= 2 else 0,
        "competitor_present": comp_present,
        "discount_requested": 1 if price_obj_count > 0 else 0,
        "proposal_sent": 1 if s_idx >= 3 else 0,
        "proposal_age_days": 4 if s_idx >= 3 else 0,
        "deal_stage": latest_stage,
        "previous_stage": prev_stage,
        "stage_duration_days": 10 + total_calls * 2,
        "customer_interest_score": 0.82 if sentiment_score > 0.6 else 0.45,
        "next_action_completed": 1,
    }
