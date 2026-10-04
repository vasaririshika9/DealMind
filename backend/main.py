import json
import os
import threading
import urllib.parse
import sys
from pathlib import Path
from typing import Optional

import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


# Path setup to reliably locate files whether script is run from project root
# or backend directory
BASE_DIR = Path(__file__).resolve().parent  # backend directory
PROJECT_ROOT = BASE_DIR.parent  # deal-intelligence-agent directory

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    try:
        from ml.explainability import get_predictor
        from ml.deal_features import extract_features_from_call_history
        from ml.data_analysis import analyze_sales_data
        from services.prediction_service import get_prediction_service
    except ImportError:
        from backend.ml.explainability import get_predictor
        from backend.ml.deal_features import extract_features_from_call_history
        from backend.ml.data_analysis import analyze_sales_data
        from backend.services.prediction_service import get_prediction_service
except Exception as _ml_err:
    print(f"[BACKEND WARNING] ML predictor/analysis import notice: {_ml_err}")
    get_predictor = None
    extract_features_from_call_history = None
    analyze_sales_data = None
    get_prediction_service = None

from services.hindsight_service import (
    recall_memory,
    retain_memory,
    close_hindsight,
)


# Load .env from backend directory, project root, or parent workspace root
for env_loc in [
    BASE_DIR / ".env",
    PROJECT_ROOT / ".env",
    PROJECT_ROOT.parent / ".env",
]:
    if env_loc.exists():
        load_dotenv(env_loc, override=True)

load_dotenv()


GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    print("[WARNING] GROQ_API_KEY is not set in your .env file.")
    print("[WARNING] The server will run, but LLM brief generation requires GROQ_API_KEY.")
else:
    print("GROQ KEY AVAILABLE:", bool(GROQ_API_KEY))
    print("GROQ MODEL:", os.getenv("GROQ_MODEL"))

GROQ_MODEL = os.getenv(
    "GROQ_MODEL",
    "openai/gpt-oss-120b"
)


DATA_FILE_PATH = (
    BASE_DIR / "data" / "mock_calls.json"
    if (BASE_DIR / "data" / "mock_calls.json").exists()
    else PROJECT_ROOT / "data" / "mock_calls.json"
)


from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

app = FastAPI(
    title="DealSight AI API",
    description="AI-Powered Sales Deal Intelligence & Prediction Assistant",
    version="1.0.0",
)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    return JSONResponse(
        status_code=400,
        content={"detail": "We couldn't generate the forecast. Please check the deal information."},
    )


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# In-memory brief cache for instant 0ms responses on cached calls
BRIEF_CACHE = {}


def get_customer_history(customer_name: str) -> list:
    """Reads data/mock_calls.json and returns all call records
    for customer_name, sorted by call_number.
    """

    data_path = (
        DATA_FILE_PATH
        if DATA_FILE_PATH.exists()
        else Path("data/mock_calls.json")
    )

    if not data_path.exists():
        raise FileNotFoundError(
            f"Mock calls file not found at {data_path}"
        )

    with open(data_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    customer_records = [
        r
        for r in records
        if r.get("customer_name", "").strip().lower()
        == customer_name.strip().lower()
    ]

    # Sort records by call_number
    customer_records.sort(
        key=lambda x: x.get("call_number", 0)
    )

    return customer_records


def generate_brief(
    customer_name: str,
    up_to_call_number: int = None,
    force_refresh: bool = False,
) -> str:
    """Gets customer call history, builds LLM prompt, calls Groq API
    (with caching for speed), and returns pre-call brief as plain text.
    """

    history = get_customer_history(customer_name)

    if up_to_call_number is not None:
        history = [
            call
            for call in history
            if call.get("call_number", 0)
            <= up_to_call_number
        ]

    if not history:
        return (
            f"No call history found for customer: "
            f"{customer_name}"
        )

    max_call_num = history[-1].get(
        "call_number",
        up_to_call_number or 0,
    )

    cache_key = (
        f"{customer_name.strip().lower()}_call_"
        f"{max_call_num}_{len(history)}"
    )

    # Check cache first for instant 0ms response
    if not force_refresh and cache_key in BRIEF_CACHE:
        return BRIEF_CACHE[cache_key]

    company_name = history[0].get(
        "company_name",
        "Unknown Company",
    )

    # Format call history details into prompt context
    history_summary = []

    for call in history:
        history_summary.append(
            f"Call #{call.get('call_number')} "
            f"({call.get('date')}):\n"
            f"  - Stage: {call.get('deal_stage')}\n"
            f"  - Sentiment: {call.get('sentiment')}\n"
            f"  - Objection Raised: "
            f"{call.get('objection_raised')}\n"
            f"  - Competitor Mentioned: "
            f"{call.get('competitor_mentioned') or 'None'}\n"
            f"  - Key Quote: "
            f"\"{call.get('key_quote')}\"\n"
            f"  - Next Step Promised: "
            f"{call.get('next_step_promised')}"
        )

    full_history_text = "\n\n".join(history_summary)

    # Query Hindsight memory service for long-term customer context
    hindsight_text = ""

    try:
        query = (
            f"What key preferences, objections, competitors, "
            f"or details are known about {customer_name} "
            f"from {company_name}?"
        )

        # Retrieve memories using customer + company tags
        recalled = recall_memory(
            query,
            tags=[customer_name, company_name],
        )

        if recalled:
            items = (
                getattr(recalled, "results", recalled)
                if not isinstance(recalled, list)
                else recalled
            )

            hindsight_facts = []

            for item in items:
                text = (
                    getattr(item, "text", None)
                    if not isinstance(item, dict)
                    else item.get("text")
                )

                if text:
                    hindsight_facts.append(
                        f"- {text.strip()}"
                    )

            if hindsight_facts:
                hindsight_text = "\n".join(
                    hindsight_facts
                )

    except Exception as e:
        print(
            f"[BACKEND WARNING] Hindsight recall failed: {e}"
        )

    hindsight_section = (
        f"\n\n--- HINDSIGHT MEMORY CONTEXT ---\n"
        f"{hindsight_text}\n"
        f"--------------------------------"
        if hindsight_text
        else ""
    )

    # Compute Machine Learning Prediction & Explainability
    prediction_info = None
    if extract_features_from_call_history and get_predictor:
        try:
            deal_feats = extract_features_from_call_history(
                customer_name=customer_name,
                company_name=company_name,
                history=history,
            )
            predictor = get_predictor()
            prediction_info = predictor.predict_deal(deal_feats)
        except Exception as e:
            print(f"[BACKEND WARNING] ML prediction failed in generate_brief: {e}")

    ml_section = ""
    if prediction_info:
        pos_facts = ", ".join([f"{f['label']} (+{f['impact']})" for f in prediction_info.get("positive_factors", [])[:3]])
        risk_facts = ", ".join([f"{f['label']} ({f['impact']})" for f in prediction_info.get("risk_factors", [])[:3]])
        ml_section = (
            f"\n\n--- PREDICTIVE MACHINE LEARNING INTELLIGENCE ---\n"
            f"- Progress Probability: {prediction_info['progress_probability']:.0%}\n"
            f"- Loss Probability: {prediction_info['loss_probability']:.0%}\n"
            f"- Deal Risk Level: {prediction_info['risk_level'].upper()} (Risk Score: {prediction_info['risk_score']}/100)\n"
            f"- Predicted Next Stage: {prediction_info['predicted_next_stage']}\n"
            f"- Top Positive Drivers: {pos_facts}\n"
            f"- Key Risk Factors: {risk_facts}\n"
            f"- Recommended Action: {prediction_info['recommended_action']}\n"
            f"--------------------------------------------------"
        )

    deal_health_str = prediction_info.get("deal_health", "Healthy Deal") if prediction_info else "Healthy Deal"
    forecast_str = prediction_info.get("forecast_summary", "Likely to move forward") if prediction_info else "Likely to move forward"
    chance_prog_str = prediction_info.get("chance_moving_forward", "82%") if prediction_info else "82%"
    chance_loss_str = prediction_info.get("chance_losing", "18%") if prediction_info else "18%"
    risk_level_str = prediction_info.get("risk_level", "Low") if prediction_info else "Low"
    risk_score_str = f"{prediction_info['risk_score']}/100" if prediction_info else "18/100"
    next_step_str = prediction_info.get("likely_next_step", "Negotiation") if prediction_info else "Negotiation"
    action_str = prediction_info.get("what_to_do_next", "Follow up with executive ROI case.") if prediction_info else "Follow up with executive ROI case."

    prompt = f"""You are DealSight AI, an intelligent B2B sales advisor. Analyze the sales interaction history and predictive forecast for customer "{customer_name}" from {company_name}:

--- CALL HISTORY ---

{full_history_text}

--------------------{hindsight_section}{ml_section}

Format your briefing in simple, clear business language. A normal salesperson must understand it within 5 seconds.
Avoid machine learning jargon (do NOT mention classifiers, algorithms, weights, or statistics).

Use this exact structure:

## DEAL HEALTH
{deal_health_str} — "{forecast_str}"
{chance_prog_str} chance of moving forward | {chance_loss_str} chance of losing
Deal Risk: {risk_level_str} Risk ({risk_score_str})
Likely Next Step: {next_step_str}

## WHY?
- [2 to 3 plain-English positive signals driving deal progress with checkmarks]
- [1 to 2 plain-English warning signs or friction points with warning signs]

## WHAT TO DO NEXT
"{action_str}"

## CUSTOMER MEMORY
- Previous concerns: [Summary of objections from previous calls]
- Past successful approach: [What worked previously or recommendations from Hindsight]
"""

    api_key = GROQ_API_KEY or os.getenv(
        "GROQ_API_KEY"
    )

    if not api_key:
        # Fallback to direct ML + Memory structured brief if Groq key is not configured
        if prediction_info:
            pos_bullets = "\n".join([f"{s}" for s in prediction_info.get("positive_signals", [])[:3]])
            warn_bullets = "\n".join([f"{w}" for w in prediction_info.get("warning_signs", [])[:2]])
            last_call = history[-1] if history else {}
            last_obj = last_call.get("objection_raised", "None")
            quote = last_call.get("key_quote", "")
            return (
                f"## DEAL HEALTH\n"
                f"{deal_health_str} — \"{forecast_str}\"\n"
                f"{chance_prog_str} chance of moving forward | {chance_loss_str} chance of losing\n"
                f"Deal Risk: {risk_level_str} Risk ({risk_score_str})\n"
                f"Likely Next Step: {next_step_str}\n\n"
                f"## WHY?\n"
                f"{pos_bullets}\n"
                f"{warn_bullets}\n\n"
                f"## WHAT TO DO NEXT\n"
                f"\"{action_str}\"\n\n"
                f"## CUSTOMER MEMORY\n"
                f"- Previous concern: {last_obj}\n"
                f"- Past successful approach: Executive value-based ROI business case and scheduling next steps within 48 hours.\n"
                f"- Key customer quote: \"{quote}\"\n"
            )
        return (
            "Error: GROQ_API_KEY environment variable "
            "is not set."
        )

    url = (
        "https://api.groq.com/openai/v1/"
        "chat/completions"
    )

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    # High-speed model priority list on Groq LPUs
    # with 20s timeout per call
    env_model = (
        os.getenv("GROQ_MODEL")
        or GROQ_MODEL
    )

    models_to_try = (
        [env_model]
        if env_model
        else []
    )

    for m in [
        "openai/gpt-oss-120b",
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
    ]:
        if m not in models_to_try:
            models_to_try.append(m)

    for model_name in models_to_try:

        payload = {
            "model": model_name,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a professional AI "
                        "Deal Intelligence Assistant."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            "temperature": 0.6,
            "max_tokens": 500,
        }

        try:
            response = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=20,
            )

            if response.status_code == 200:
                result_json = response.json()

                brief_text = (
                    result_json["choices"][0]
                    ["message"]["content"]
                )

                BRIEF_CACHE[cache_key] = brief_text

                return brief_text

            else:
                print(
                    f"[BACKEND WARNING] Model "
                    f"{model_name} returned status "
                    f"{response.status_code}"
                )

        except Exception as e:
            print(
                f"[BACKEND WARNING] Model "
                f"{model_name} failed or timed out: {e}"
            )
            continue

    return (
        "Error generating brief: Groq API attempt "
        "timed out or failed."
    )


def save_call(call_data: dict) -> None:
    """Appends a new call record dictionary to
    data/mock_calls.json and saves the updated list.
    """

    global BRIEF_CACHE

    BRIEF_CACHE.clear()

    data_path = (
        DATA_FILE_PATH
        if DATA_FILE_PATH.exists()
        else Path("data/mock_calls.json")
    )

    records = []

    if data_path.exists():
        with open(data_path, "r", encoding="utf-8") as f:
            records = json.load(f)

    # Update if call record with same customer_name
    # & call_number exists, otherwise append
    existing_index = None

    for idx, r in enumerate(records):

        if (
            r.get("customer_name", "").strip().lower()
            == call_data.get(
                "customer_name", ""
            ).strip().lower()
            and r.get("call_number")
            == call_data.get("call_number")
        ):
            existing_index = idx
            break

    if existing_index is not None:
        records[existing_index] = call_data
    else:
        records.append(call_data)

    with open(data_path, "w", encoding="utf-8") as f:
        json.dump(
            records,
            f,
            indent=2,
            ensure_ascii=False,
        )

    # Safely retain call insights memory statement in Hindsight
    try:

        c_name = call_data.get(
            "customer_name",
            "Unknown Customer",
        )

        c_company = call_data.get(
            "company_name",
            "Unknown Company",
        )

        c_objection = call_data.get(
            "objection_raised",
            "None",
        )

        c_competitor = (
            call_data.get(
                "competitor_mentioned"
            )
            or "None"
        )

        c_sentiment = call_data.get(
            "sentiment",
            "neutral",
        )

        c_stage = call_data.get(
            "deal_stage",
            "negotiation",
        )

        c_quote = call_data.get(
            "key_quote",
            "",
        )

        c_next = call_data.get(
            "next_step_promised",
            "",
        )

        memory_statement = (
            f"Customer {c_name} from {c_company} "
            f"(Stage: {c_stage}, "
            f"Sentiment: {c_sentiment}). "
            f"Objection raised: {c_objection}. "
            f"Competitor mentioned: {c_competitor}. "
            f"Key quote: \"{c_quote}\". "
            f"Next step: {c_next}."
        )

        # Store the memory using customer + company tags
        retain_memory(
            memory_statement,
            tags=[c_name, c_company],
        )

    except Exception as e:
        print(
            f"[BACKEND WARNING] Hindsight retain failed: {e}"
        )


def prewarm_cache_bg():
    """Asynchronously pre-populates brief cache in background
    so UI loads instantly.
    """

    try:

        customers = [
            "Rahul Sharma",
            "Priya Nair",
            "Arjun Mehta",
            "Sneha Kulkarni",
        ]

        for cust in customers:

            try:

                history = get_customer_history(cust)

                for call in history:

                    c_num = call.get(
                        "call_number",
                        1,
                    )

                    generate_brief(
                        cust,
                        c_num,
                    )

            except Exception:
                pass

    except Exception:
        pass


@app.on_event("startup")
def startup_event():
    """Triggers background cache prewarming when FastAPI starts."""

    threading.Thread(
        target=prewarm_cache_bg,
        daemon=True,
    ).start()


@app.on_event("shutdown")
def shutdown_event():
    """Closes the Hindsight client when FastAPI shuts down."""

    close_hindsight()


# --- FASTAPI ENDPOINTS ---

@app.get("/customers")
def list_customers():
    """Returns list of unique customers with details."""

    data_path = (
        DATA_FILE_PATH
        if DATA_FILE_PATH.exists()
        else Path("data/mock_calls.json")
    )

    if not data_path.exists():
        return []

    with open(data_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    customers_dict = {}

    for r in records:

        name = r.get("customer_name")

        if name and name not in customers_dict:

            customers_dict[name] = {
                "customer_name": name,
                "company_name": r.get(
                    "company_name",
                    "",
                ),
                "total_calls": 0,
                "latest_sentiment": r.get(
                    "sentiment",
                    "neutral",
                ),
                "latest_stage": r.get(
                    "deal_stage",
                    "",
                ),
            }

        if name in customers_dict:

            customers_dict[name]["total_calls"] += 1

            customers_dict[name]["latest_sentiment"] = r.get(
                "sentiment",
                customers_dict[name][
                    "latest_sentiment"
                ],
            )

            customers_dict[name]["latest_stage"] = r.get(
                "deal_stage",
                customers_dict[name][
                    "latest_stage"
                ],
            )

    return list(customers_dict.values())


@app.get("/calls/{customer_name}")
def get_customer_calls(customer_name: str):
    """Returns call history records for specified customer."""

    decoded_name = urllib.parse.unquote(
        customer_name
    ).strip()

    try:

        return get_customer_history(
            decoded_name
        )

    except FileNotFoundError:

        raise HTTPException(
            status_code=404,
            detail="Mock calls database not found.",
        )


@app.get(
    "/brief/{customer_name}/{up_to_call_number}"
)
def get_brief_endpoint(
    customer_name: str,
    up_to_call_number: int,
    refresh: bool = False,
):
    """Returns AI generated brief for customer up to
    call number along with call history.
    """

    decoded_customer_name = urllib.parse.unquote(
        customer_name
    ).strip()

    print(
        f"[BACKEND] Received /brief request for "
        f"'{decoded_customer_name}' up to "
        f"call #{up_to_call_number}"
    )

    try:

        history = get_customer_history(
            decoded_customer_name
        )

    except FileNotFoundError as e:

        print(
            f"[BACKEND ERROR] FileNotFoundError for "
            f"'{decoded_customer_name}': {e}"
        )

        raise HTTPException(
            status_code=404,
            detail=(
                f"Mock calls database not found for "
                f"'{decoded_customer_name}'."
            ),
        )

    except Exception as e:

        print(
            f"[BACKEND ERROR] Error fetching history: {e}"
        )

        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch history: {str(e)}",
        )

    calls_up_to = [
        c
        for c in history
        if c.get("call_number", 0)
        <= up_to_call_number
    ]

    if not calls_up_to:

        print(
            f"[BACKEND ERROR] No calls up to "
            f"#{up_to_call_number} found for "
            f"'{decoded_customer_name}'"
        )

        raise HTTPException(
            status_code=404,
            detail=(
                f"No call history up to "
                f"call #{up_to_call_number} "
                f"for customer "
                f"{decoded_customer_name}"
            ),
        )

    try:

        brief_text = generate_brief(
            decoded_customer_name,
            up_to_call_number,
            force_refresh=refresh,
        )

        if brief_text.startswith(
            "Error generating brief"
        ):

            print(
                f"[BACKEND ERROR] Brief generation error: "
                f"{brief_text}"
            )

            raise HTTPException(
                status_code=500,
                detail=brief_text,
            )

        print(
            f"[BACKEND] Successfully generated brief for "
            f"'{decoded_customer_name}' "
            f"call #{up_to_call_number}"
        )

        pred_data = None
        if extract_features_from_call_history and get_predictor:
            try:
                feats = extract_features_from_call_history(
                    customer_name=decoded_customer_name,
                    company_name=calls_up_to[0].get("company_name", ""),
                    history=calls_up_to,
                )
                pred_data = get_predictor().predict_deal(feats)
            except Exception as e:
                print(f"[BACKEND WARNING] Failed to compute prediction in get_brief_endpoint: {e}")

        return {
            "customer_name": decoded_customer_name,
            "up_to_call_number": up_to_call_number,
            "brief": brief_text,
            "history": history,
            "current_call": (
                calls_up_to[-1]
                if calls_up_to
                else None
            ),
            "total_calls_available": len(history),
            "prediction": pred_data,
        }

    except HTTPException:
        raise

    except Exception as e:

        print(
            f"[BACKEND ERROR] Unhandled exception in "
            f"generate_brief: {e}"
        )

        raise HTTPException(
            status_code=500,
            detail=(
                f"Failed to generate brief: "
                f"{str(e)}"
            ),
        )


class CallRecordInput(BaseModel):
    customer_name: str
    company_name: str
    call_number: int
    date: str
    objection_raised: str
    competitor_mentioned: Optional[str] = None
    sentiment: str
    deal_stage: str
    key_quote: str
    next_step_promised: str


@app.post("/calls")
def create_call(call: CallRecordInput):
    """Saves a new call record."""

    save_call(
        call.model_dump()
    )

    return {
        "status": "success",
        "message": (
            f"Call #{call.call_number} "
            f"saved for {call.customer_name}"
        ),
    }


class SaveCallPayload(BaseModel):
    customer_name: str
    company_name: Optional[str] = None
    call_number: Optional[int] = None
    date: Optional[str] = None
    objection_raised: str
    competitor_mentioned: Optional[str] = None
    sentiment: str
    deal_stage: Optional[str] = "negotiation"
    key_quote: str
    next_step_promised: str


@app.post("/save-call")
def save_call_endpoint(
    payload: SaveCallPayload,
):
    """POST endpoint /save-call to append a new sales
    call record using save_call().
    """

    history = []

    try:

        history = get_customer_history(
            payload.customer_name
        )

    except FileNotFoundError:
        pass

    company_name = (
        payload.company_name
        or (
            history[0].get("company_name")
            if history
            else "Unknown Company"
        )
    )

    call_number = (
        payload.call_number
        or (
            len(history) + 1
            if history
            else 1
        )
    )

    call_date = (
        payload.date
        or "2026-09-27"
    )

    call_dict = {
        "customer_name": payload.customer_name,
        "company_name": company_name,
        "call_number": int(call_number),
        "date": call_date,
        "objection_raised": payload.objection_raised,
        "competitor_mentioned": (
            payload.competitor_mentioned
            if payload.competitor_mentioned
            else None
        ),
        "sentiment": payload.sentiment,
        "deal_stage": (
            payload.deal_stage
            or "negotiation"
        ),
        "key_quote": payload.key_quote,
        "next_step_promised": payload.next_step_promised,
    }

    save_call(call_dict)

    print(
        f"[BACKEND] Call #{call_number} saved for "
        f"'{payload.customer_name}'"
    )

    return {
        "status": "success",
        "message": (
            f"Call #{call_number} saved for "
            f"{payload.customer_name}"
        ),
        "call": call_dict,
    }


# ========================================================
# --- PREDICTIVE ANALYTICS & EXPLAINABLE AI ENDPOINTS ---
# ========================================================

class DealPredictionRequest(BaseModel):
    deal_value: Optional[float] = 50000.0
    total_calls: Optional[int] = 3
    days_since_last_call: Optional[int] = 3
    calls_last_7_days: Optional[int] = 1
    calls_last_30_days: Optional[int] = 3
    price_objections: Optional[int] = 0
    competitor_mentions: Optional[int] = 0
    demo_requested: Optional[int] = 1
    demo_completed: Optional[int] = 0
    decision_maker_present: Optional[int] = 0
    decision_maker_engagement: Optional[float] = 0.0
    followups: Optional[int] = 2
    followup_response_rate: Optional[float] = 0.70
    response_delay_hours: Optional[float] = 12.0
    sentiment_score: Optional[float] = 0.65
    sentiment_trend: Optional[int] = 0
    engagement_score: Optional[float] = 0.60
    objection_count: Optional[int] = 0
    objection_resolution_rate: Optional[float] = 1.0
    budget_confirmed: Optional[int] = 0
    timeline_confirmed: Optional[int] = 0
    competitor_present: Optional[int] = 0
    discount_requested: Optional[int] = 0
    proposal_sent: Optional[int] = 0
    proposal_age_days: Optional[int] = 0
    deal_stage: Optional[str] = "Evaluation"
    previous_stage: Optional[str] = "Demo"
    stage_duration_days: Optional[int] = 14
    customer_interest_score: Optional[float] = 0.70
    next_action_completed: Optional[int] = 1
    industry: Optional[str] = "Technology"
    company_size: Optional[str] = "Mid-Market"


@app.post("/predict-deal")
def predict_deal_endpoint(payload: DealPredictionRequest):
    """POST /predict-deal: Evaluates a deal vector using the best ML model,
    returning probability of progression vs loss, risk score, factors, and recommended action.
    """
    if not get_predictor:
        raise HTTPException(
            status_code=503,
            detail="Machine learning predictor is not available. Please ensure model is trained.",
        )

    try:
        predictor = get_predictor()
        deal_dict = payload.dict()
        result = predictor.predict_deal(deal_dict)
        return result
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate prediction: {str(e)}"
        )


class PredictRequest(BaseModel):
    total_calls: Optional[int] = 4
    days_since_last_call: Optional[int] = 2
    price_objections: Optional[int] = 1
    competitor_mentions: Optional[int] = 0
    demo_requested: Optional[int] = 1
    decision_maker_present: Optional[int] = 1
    followups: Optional[int] = 3
    sentiment_score: Optional[float] = 0.78
    deal_value: Optional[float] = 50000.0
    response_delay_hours: Optional[float] = 12.0
    engagement_score: Optional[float] = 0.65
    deal_stage: Optional[str] = "Evaluation"


@app.post("/predict")
def predict_endpoint(payload: PredictRequest):
    """POST /predict: Evaluates deal input features using the trained ML model.
    Returns:
    - Structured Technical Info: prediction, progress_probability, loss_probability, risk_level, important_factors, recommended_action
    - User-Friendly Translations: headline, probability_text, risk_text, why, concerns, next_action
    """
    if not get_prediction_service:
        # Fallback to get_predictor if prediction_service unavailable
        if get_predictor:
            try:
                predictor = get_predictor()
                data = payload.dict()
                raw_res = predictor.predict_deal(data)
                return {
                    "prediction": raw_res["prediction"],
                    "progress_probability": raw_res["progress_probability"],
                    "loss_probability": raw_res["loss_probability"],
                    "risk_level": raw_res["risk_level"],
                    "important_factors": raw_res.get("positive_signals", []),
                    "recommended_action": raw_res["recommended_action"],
                    "headline": raw_res["forecast_summary"],
                    "probability_text": f"{raw_res['chance_moving_forward']} chance of moving forward",
                    "risk_text": f"{raw_res['risk_level']} risk",
                    "why": [s.replace("✓ ", "") for s in raw_res.get("positive_signals", [])],
                    "concerns": [w.replace("⚠ ", "") for w in raw_res.get("warning_signs", [])],
                    "next_action": raw_res["what_to_do_next"],
                }
            except Exception as e:
                raise HTTPException(
                    status_code=500,
                    detail="We couldn't generate the forecast. Please check the deal information."
                )
        raise HTTPException(
            status_code=503,
            detail="We couldn't generate the forecast. The prediction service is initializing."
        )

    try:
        service = get_prediction_service()
        input_dict = payload.dict()
        result = service.predict(input_dict)
        return result
    except Exception as e:
        print(f"[BACKEND ERROR] /predict failed: {e}")
        raise HTTPException(
            status_code=500,
            detail="We couldn't generate the forecast. Please check the deal information."
        )


@app.get("/prediction/{customer_or_deal_id}")
def get_deal_prediction_endpoint(customer_or_deal_id: str):
    """GET /prediction/{customer_or_deal_id}: Computes ML prediction for a known customer
    from historical call interactions or defaults.
    """
    decoded = urllib.parse.unquote(customer_or_deal_id).strip()

    if not get_predictor:
        raise HTTPException(
            status_code=503,
            detail="Machine learning predictor is not initialized.",
        )

    try:
        history = []
        company_name = "Enterprise Client"
        try:
            history = get_customer_history(decoded)
            if history:
                company_name = history[0].get("company_name", company_name)
        except Exception:
            pass

        if extract_features_from_call_history:
            feats = extract_features_from_call_history(
                customer_name=decoded,
                company_name=company_name,
                history=history,
            )
        else:
            feats = {"customer": decoded, "company_name": company_name}

        if get_prediction_service:
            service = get_prediction_service()
            res = service.predict(feats)
        else:
            predictor = get_predictor()
            res = predictor.predict_deal(feats)

        res["customer_name"] = decoded
        res["company_name"] = company_name
        res["total_historical_calls"] = len(history)
        return res
    except Exception as e:
        print(f"[BACKEND ERROR] /prediction/{customer_or_deal_id} failed: {e}")
        raise HTTPException(
            status_code=500,
            detail="We couldn't generate the forecast. Please check the deal information."
        )


@app.get("/model-metrics")
def get_model_metrics_endpoint():
    """GET /model-metrics: Returns multi-model comparison table, holdout test metrics,
    and confusion matrix.
    """
    model_dir = BASE_DIR / "models" if (BASE_DIR / "models").exists() else PROJECT_ROOT / "models"
    metrics_path = model_dir / "metrics.json"
    eval_path = model_dir / "evaluation_report.json"

    if not metrics_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Model metrics not found. Please run training pipeline first."
        )

    with open(metrics_path, "r") as f:
        metrics_data = json.load(f)

    if eval_path.exists():
        with open(eval_path, "r") as f:
            eval_data = json.load(f)
            metrics_data["holdout_evaluation"] = eval_data

    next_stage_meta_path = model_dir / "next_stage_metadata.json"
    if next_stage_meta_path.exists():
        with open(next_stage_meta_path, "r") as f:
            metrics_data["next_stage_model"] = json.load(f)

    next_stage_comp_path = model_dir / "next_stage_comparison.json"
    if next_stage_comp_path.exists():
        with open(next_stage_comp_path, "r") as f:
            metrics_data["next_stage_comparison"] = json.load(f)

    return metrics_data


@app.get("/feature-importance")
def get_feature_importance_endpoint():
    """GET /feature-importance: Returns top predictive feature weights driving deal progression."""
    model_dir = BASE_DIR / "models" if (BASE_DIR / "models").exists() else PROJECT_ROOT / "models"
    feat_path = model_dir / "feature_importance.json"

    if not feat_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Feature importance data not found. Please run training pipeline first."
        )

    with open(feat_path, "r") as f:
        features_data = json.load(f)

    return {"top_features": features_data}


@app.get("/dataset-insights")
@app.get("/sales-insights")
def get_dataset_insights_endpoint():
    """GET /dataset-insights: Returns dynamically calculated historical dataset summary cards
    (Total Deals, Outcomes, Avg Calls, Objections, Follow-ups) and the 4 business charts.
    """
    if not analyze_sales_data:
        raise HTTPException(
            status_code=503,
            detail="Dataset analysis engine is not available."
        )

    try:
        data = analyze_sales_data()
        return data
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error analyzing dataset: {str(e)}"
        )


if __name__ == "__main__":

    import uvicorn

    # Ensure backend folder is in Python path
    backend_dir = Path(__file__).resolve().parent

    if str(backend_dir) not in sys.path:
        sys.path.insert(0, str(backend_dir))

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(
            encoding="utf-8"
        )

    print(
        "Starting FastAPI server on "
        "http://localhost:8001 ..."
    )

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8001,
    )