"""
DealSight AI - Historical Sales Dataset Generator
Generates realistic historical deal snapshots for ML training & evaluation.
Relationships reflect real-world B2B sales dynamics with realistic noise.
"""

import os
import random
import numpy as np
import pandas as pd

# Set seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

INDUSTRIES = [
    "Technology",
    "Healthcare",
    "Manufacturing",
    "Retail",
    "Financial Services",
    "Logistics",
    "Education",
]

COMPANY_SIZES = ["Startup", "SMB", "Mid-Market", "Enterprise"]

STAGES = ["Discovery", "Demo", "Evaluation", "Negotiation", "Proposal"]

CUSTOMERS = [
    "Zenith Textiles", "Coastal Foods Ltd", "BrightPath Logistics", "Everline Retail",
    "Apex Cloud Systems", "NexGen Health", "Summit BioPharma", "Pulse Dynamics",
    "Vanguard Supply", "Trident Tech", "Solstice Media", "Frontier FinCorp",
    "BlueShift AI", "Acuity Networks", "Cobalt Freight", "Prime Packaging",
    "Cascade Security", "Beacon Robotics", "Atlas Energy", "Veritas Analytics",
    "Helix Diagnostics", "Skyward Aerospace", "OmniCommerce", "Novus Capital",
    "Aegis Software", "Kestrel Instruments", "Synthetix Labs", "Vector Mobility"
]

def generate_sales_dataset(n_samples: int = 2500, output_path: str = "data/sales_data.csv") -> pd.DataFrame:
    """Generate n_samples realistic sales deal records."""
    records = []

    for i in range(1, n_samples + 1):
        deal_id = f"DEAL-{1000 + i}"
        customer = random.choice(CUSTOMERS)
        industry = random.choice(INDUSTRIES)
        company_size = random.choices(COMPANY_SIZES, weights=[0.25, 0.35, 0.25, 0.15])[0]

        # Deal value varies by company size
        if company_size == "Startup":
            deal_value = round(random.uniform(8000, 35000), -2)
        elif company_size == "SMB":
            deal_value = round(random.uniform(25000, 75000), -2)
        elif company_size == "Mid-Market":
            deal_value = round(random.uniform(60000, 160000), -2)
        else: # Enterprise
            deal_value = round(random.uniform(120000, 350000), -2)

        # Call activity
        total_calls = random.randint(1, 14)
        days_since_last_call = random.randint(0, 45)
        calls_last_7_days = min(total_calls, random.randint(0, min(total_calls, 4)))
        calls_last_30_days = min(total_calls, calls_last_7_days + random.randint(0, min(total_calls, 7)))

        # Deal stage & progression
        stage_idx = random.randint(0, len(STAGES) - 1)
        deal_stage = STAGES[stage_idx]
        previous_stage = STAGES[max(0, stage_idx - 1)] if stage_idx > 0 else "Lead"
        stage_duration_days = random.randint(2, 60)

        # Engagement & signals
        demo_requested = 1 if stage_idx >= 1 or random.random() > 0.4 else 0
        demo_completed = 1 if demo_requested and (stage_idx >= 1 and random.random() > 0.25) else 0

        decision_maker_present = 1 if (company_size in ["Startup", "SMB"] and random.random() > 0.35) or (random.random() > 0.45) else 0
        decision_maker_engagement = round(random.uniform(0.4, 0.98), 2) if decision_maker_present else round(random.uniform(0.0, 0.3), 2)

        followups = random.randint(1, 9)
        followup_response_rate = round(random.uniform(0.15, 0.95), 2)
        response_delay_hours = round(max(1.0, np.random.exponential(scale=24.0)), 1)
        if response_delay_hours > 120.0:
            response_delay_hours = 120.0

        # Sentiment & objections
        sentiment_score = round(random.uniform(0.15, 0.92), 2)
        sentiment_trend = random.choice([-1, 0, 1])
        engagement_score = round(random.uniform(0.2, 0.98), 2)
        customer_interest_score = round(random.uniform(0.25, 0.99), 2)

        objection_count = random.randint(0, 5)
        price_objections = min(objection_count, random.randint(0, min(objection_count, 3)))
        objection_resolution_rate = round(random.uniform(0.0, 1.0), 2) if objection_count > 0 else 1.0

        # Competitor & commercial terms
        competitor_present = 1 if random.random() > 0.55 else 0
        competitor_mentions = random.randint(1, 3) if competitor_present else 0

        budget_confirmed = 1 if (stage_idx >= 2 and random.random() > 0.35) or random.random() > 0.6 else 0
        timeline_confirmed = 1 if (stage_idx >= 2 and random.random() > 0.4) or random.random() > 0.55 else 0
        discount_requested = 1 if price_objections > 0 and random.random() > 0.3 else (1 if random.random() > 0.65 else 0)

        proposal_sent = 1 if stage_idx >= 3 or random.random() > 0.7 else 0
        proposal_age_days = random.randint(1, 35) if proposal_sent else 0
        next_action_completed = 1 if random.random() > 0.35 else 0

        # --- Latent Outcome Score Calculation ---
        # Business logic driving deal progression probability:
        latent_score = (
            + 1.70 * decision_maker_present
            + 1.45 * decision_maker_engagement
            + 1.40 * budget_confirmed
            + 1.25 * timeline_confirmed
            + 1.50 * demo_completed
            + 1.85 * engagement_score
            + 1.40 * customer_interest_score
            + 1.55 * sentiment_score
            + 0.75 * sentiment_trend
            + 1.30 * objection_resolution_rate
            + 0.85 * next_action_completed
            + 1.10 * followup_response_rate
            + (0.35 * stage_idx) # later stages generally progressed further
            - 1.65 * price_objections
            - 1.15 * competitor_mentions
            - 1.30 * (response_delay_hours / 45.0)
            - 1.20 * (days_since_last_call / 20.0)
            - 1.05 * (stage_duration_days / 30.0)
            - 0.85 * discount_requested
            - 0.90 * competitor_present
            - 1.10 * (proposal_age_days / 20.0 if proposal_sent else 0)
        )

        # Baseline offset to balance classes (~54% Progressed, 46% Lost)
        baseline_offset = -3.8

        # Add realistic stochastic noise so the model isn't deterministic
        noise = np.random.normal(loc=0.0, scale=1.35)
        total_z = latent_score + baseline_offset + noise

        # Logistic sigmoid probability
        prob_progress = 1.0 / (1.0 + np.exp(-total_z))

        deal_outcome = "Progressed" if prob_progress >= 0.5 else "Lost"

        records.append({
            "deal_id": deal_id,
            "customer": customer,
            "industry": industry,
            "company_size": company_size,
            "deal_value": deal_value,
            "total_calls": total_calls,
            "days_since_last_call": days_since_last_call,
            "calls_last_7_days": calls_last_7_days,
            "calls_last_30_days": calls_last_30_days,
            "price_objections": price_objections,
            "competitor_mentions": competitor_mentions,
            "demo_requested": demo_requested,
            "demo_completed": demo_completed,
            "decision_maker_present": decision_maker_present,
            "decision_maker_engagement": decision_maker_engagement,
            "followups": followups,
            "followup_response_rate": followup_response_rate,
            "response_delay_hours": response_delay_hours,
            "sentiment_score": sentiment_score,
            "sentiment_trend": sentiment_trend,
            "engagement_score": engagement_score,
            "objection_count": objection_count,
            "objection_resolution_rate": objection_resolution_rate,
            "budget_confirmed": budget_confirmed,
            "timeline_confirmed": timeline_confirmed,
            "competitor_present": competitor_present,
            "discount_requested": discount_requested,
            "proposal_sent": proposal_sent,
            "proposal_age_days": proposal_age_days,
            "deal_stage": deal_stage,
            "previous_stage": previous_stage,
            "stage_duration_days": stage_duration_days,
            "customer_interest_score": customer_interest_score,
            "next_action_completed": next_action_completed,
            "deal_outcome": deal_outcome,
        })

    df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[DATASET GENERATOR] Successfully generated {len(df)} records at: {output_path}")
    print(f"[DATASET GENERATOR] Outcome distribution:\n{df['deal_outcome'].value_counts(normalize=True).round(3)}")
    return df

if __name__ == "__main__":
    generate_sales_dataset(n_samples=2500, output_path="data/sales_data.csv")
