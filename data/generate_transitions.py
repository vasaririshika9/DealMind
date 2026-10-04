"""
DealSight AI - Historical Deal Stage Transition Generator
Generates realistic multi-interaction sequential sales records.
Derives observed 'next_stage' strictly by shifting chronological call stages per deal.
Zero target leakage: future call information and final deal outcomes are strictly excluded.
"""

import os
import random
from pathlib import Path
import numpy as np
import pandas as pd

# Set seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

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

CUSTOMERS = [
    "Zenith Textiles", "Coastal Foods Ltd", "BrightPath Logistics", "Everline Retail",
    "Apex Cloud Systems", "NexGen Health", "Summit BioPharma", "Pulse Dynamics",
    "Vanguard Supply", "Trident Tech", "Solstice Media", "Frontier FinCorp",
    "BlueShift AI", "Acuity Networks", "Cobalt Freight", "Prime Packaging",
    "Cascade Security", "Beacon Robotics", "Atlas Energy", "Veritas Analytics",
    "Helix Diagnostics", "Skyward Aerospace", "OmniCommerce", "Novus Capital",
    "Aegis Software", "Kestrel Instruments", "Synthetix Labs", "Vector Mobility"
]

STAGES_ORDER = ["Discovery", "Demo", "Evaluation", "Negotiation", "Proposal"]

def generate_deal_transitions(
    n_deals: int = 1200,
    output_path: str = "data/deal_transitions.csv"
) -> pd.DataFrame:
    """Simulates realistic chronological interactions per deal, then derives actual next_stage."""
    raw_calls = []

    for d_idx in range(1, n_deals + 1):
        deal_id = f"D{d_idx:04d}"
        customer = random.choice(CUSTOMERS)
        industry = random.choice(INDUSTRIES)
        company_size = random.choices(COMPANY_SIZES, weights=[0.25, 0.35, 0.25, 0.15])[0]

        if company_size == "Startup":
            deal_value = round(random.uniform(8000, 35000), -2)
        elif company_size == "SMB":
            deal_value = round(random.uniform(25000, 75000), -2)
        elif company_size == "Mid-Market":
            deal_value = round(random.uniform(60000, 160000), -2)
        else:
            deal_value = round(random.uniform(120000, 350000), -2)

        # Baseline latent momentum of this customer account
        account_quality = random.uniform(0.2, 0.8)

        # Sequential call generation
        current_stage = "Discovery"
        stage_idx = 0
        call_num = 1
        max_calls = random.randint(3, 8)
        
        # Tracking features as they evolve over calls
        demo_requested = 0
        demo_completed = 0
        decision_maker_present = 0
        cumulative_objections = 0
        sentiment = round(random.uniform(0.50, 0.80) if account_quality > 0.5 else random.uniform(0.35, 0.65), 2)
        competitor_mentions = 0

        while call_num <= max_calls:
            # Simulate features observed at call_num
            days_since_last = random.randint(1, 14) if call_num > 1 else random.randint(1, 5)
            followups = random.randint(1, 4)
            response_delay = round(max(2.0, np.random.exponential(scale=18.0 / (account_quality + 0.2))), 1)

            # Evolve signals
            if current_stage == "Discovery" and random.random() < (0.4 + 0.4 * account_quality):
                demo_requested = 1

            if current_stage == "Demo" or (demo_requested and call_num >= 2):
                if random.random() < (0.5 + 0.35 * account_quality):
                    demo_completed = 1

            if random.random() < (0.3 + 0.45 * account_quality):
                decision_maker_present = 1

            # Objection pressure
            if random.random() < (0.45 - 0.25 * account_quality):
                cumulative_objections = min(4, cumulative_objections + random.choice([0, 1]))

            # Competitor mentions
            if random.random() < 0.25:
                competitor_mentions = min(3, competitor_mentions + 1)

            # Sentiment drift
            sentiment_delta = random.uniform(-0.10, 0.12) if account_quality > 0.5 else random.uniform(-0.15, 0.08)
            sentiment = round(max(0.12, min(0.96, sentiment + sentiment_delta)), 2)
            engagement_score = round(max(0.15, min(0.98, sentiment * 0.6 + account_quality * 0.4)), 2)

            raw_calls.append({
                "deal_id": deal_id,
                "customer": customer,
                "industry": industry,
                "company_size": company_size,
                "deal_value": deal_value,
                "call_number": call_num,
                "current_stage": current_stage,
                "sentiment_score": sentiment,
                "price_objections": cumulative_objections,
                "competitor_mentions": competitor_mentions,
                "demo_requested": demo_requested,
                "demo_completed": demo_completed,
                "decision_maker_present": decision_maker_present,
                "followups": followups,
                "days_since_last_call": days_since_last,
                "response_delay_hours": response_delay,
                "engagement_score": engagement_score,
            })

            # Determine next stage transition based on current interaction signals
            # 1. Check for dropout / deal loss
            risk_signal = (
                (1.0 - sentiment) * 1.5 +
                cumulative_objections * 0.4 +
                (1 - decision_maker_present) * 0.3 +
                competitor_mentions * 0.3 +
                (response_delay / 48.0) * 0.4
            )
            
            # Chance of deal falling through to Lost
            if risk_signal > 1.8 and random.random() < 0.35:
                # Terminal transition to Lost
                current_stage = "Lost"
                raw_calls.append({
                    "deal_id": deal_id,
                    "customer": customer,
                    "industry": industry,
                    "company_size": company_size,
                    "deal_value": deal_value,
                    "call_number": call_num + 1,
                    "current_stage": "Lost",
                    "sentiment_score": max(0.1, sentiment - 0.2),
                    "price_objections": cumulative_objections,
                    "competitor_mentions": competitor_mentions,
                    "demo_requested": demo_requested,
                    "demo_completed": demo_completed,
                    "decision_maker_present": decision_maker_present,
                    "followups": followups,
                    "days_since_last_call": days_since_last + 7,
                    "response_delay_hours": response_delay + 24,
                    "engagement_score": max(0.1, engagement_score - 0.2),
                })
                break

            # 2. Check for stage progression
            progression_score = (
                sentiment * 1.4 +
                demo_completed * 0.8 +
                decision_maker_present * 0.7 +
                engagement_score * 0.8 -
                cumulative_objections * 0.35 -
                (response_delay / 48.0) * 0.3
            )

            if progression_score > 1.4:
                # Advance stage
                if stage_idx < len(STAGES_ORDER) - 1:
                    stage_idx += 1
                    current_stage = STAGES_ORDER[stage_idx]
                else:
                    # At Proposal stage: potential close
                    if random.random() < 0.65:
                        current_stage = "Closed Won"
                        raw_calls.append({
                            "deal_id": deal_id,
                            "customer": customer,
                            "industry": industry,
                            "company_size": company_size,
                            "deal_value": deal_value,
                            "call_number": call_num + 1,
                            "current_stage": "Closed Won",
                            "sentiment_score": min(0.98, sentiment + 0.1),
                            "price_objections": max(0, cumulative_objections - 1),
                            "competitor_mentions": competitor_mentions,
                            "demo_requested": 1,
                            "demo_completed": 1,
                            "decision_maker_present": 1,
                            "followups": followups,
                            "days_since_last_call": 2,
                            "response_delay_hours": 4.0,
                            "engagement_score": 0.95,
                        })
                        break
            elif progression_score < 0.7 and random.random() < 0.25:
                # Deal stalls and closes as Lost
                current_stage = "Lost"
                raw_calls.append({
                    "deal_id": deal_id,
                    "customer": customer,
                    "industry": industry,
                    "company_size": company_size,
                    "deal_value": deal_value,
                    "call_number": call_num + 1,
                    "current_stage": "Lost",
                    "sentiment_score": max(0.1, sentiment - 0.15),
                    "price_objections": cumulative_objections,
                    "competitor_mentions": competitor_mentions,
                    "demo_requested": demo_requested,
                    "demo_completed": demo_completed,
                    "decision_maker_present": decision_maker_present,
                    "followups": followups,
                    "days_since_last_call": days_since_last + 10,
                    "response_delay_hours": response_delay + 30,
                    "engagement_score": max(0.1, engagement_score - 0.2),
                })
                break
            else:
                # Stay in current stage for another call (e.g. follow-up demo, deeper evaluation)
                pass

            call_num += 1

    raw_df = pd.DataFrame(raw_calls)

    # Sort strictly chronologically by deal and call number
    raw_df = raw_df.sort_values(by=["deal_id", "call_number"]).reset_index(drop=True)

    # DERIVE NEXT STAGE TARGET BY SHIFTING FORWARD WITHIN EACH DEAL
    # This reflects what stage the deal actually entered on the next interaction
    raw_df["next_stage"] = raw_df.groupby("deal_id")["current_stage"].shift(-1)

    # Drop the final observation of each deal (where next stage was unobserved)
    transitions_df = raw_df.dropna(subset=["next_stage"]).reset_index(drop=True)

    # Filter out transitions starting from terminal states (terminal states have no forward transitions)
    transitions_df = transitions_df[~transitions_df["current_stage"].isin(["Closed Won", "Lost"])].reset_index(drop=True)

    full_output = PROJECT_ROOT / output_path
    full_output.parent.mkdir(parents=True, exist_ok=True)
    transitions_df.to_csv(full_output, index=False)

    print(f"[STAGE TRANSITION GENERATOR] Generated {len(transitions_df)} observed transitions across {transitions_df['deal_id'].nunique()} deals.")
    print(f"[STAGE TRANSITION GENERATOR] Target distribution for 'next_stage':\n{transitions_df['next_stage'].value_counts()}")
    return transitions_df

if __name__ == "__main__":
    generate_deal_transitions()
