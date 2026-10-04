import os
import sys
import aiohttp
from dotenv import load_dotenv
from hindsight_client import Hindsight

from pathlib import Path

# 1. Load the .env file using python-dotenv
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
for env_path in [BASE_DIR / ".env", PROJECT_ROOT / ".env"]:
    if env_path.exists():
        load_dotenv(env_path)
load_dotenv()

# 2. Read environment variables
api_key = os.getenv("HINDSIGHT_API_KEY")
base_url = os.getenv("HINDSIGHT_BASE_URL", "https://api.hindsight.vectorize.io")
bank_id = os.getenv("HINDSIGHT_BANK_ID", "dealsight-sales")

if not api_key:
    print("Error: HINDSIGHT_API_KEY is missing. Add it to your .env file.")
    sys.exit(1)

# Helper function to ensure HINDSIGHT_API_KEY is never printed
def sanitize_msg(msg: str) -> str:
    if api_key and api_key in msg:
        return msg.replace(api_key, "[REDACTED]")
    return msg

# 3. Initialize the official Python Hindsight client using environment variables
try:
    client = Hindsight(
        base_url=base_url,
        api_key=api_key
    )
    print("Hindsight client initialized successfully")
except Exception as e:
    err_str = sanitize_msg(str(e))
    print(f"Error initializing Hindsight client: {err_str}")
    sys.exit(1)

# Bank creation & existence handling section
try:
    bank_exists = False
    try:
        client.get_bank_config(bank_id=bank_id)
        bank_exists = True
    except aiohttp.ClientResponseError as req_err:
        if req_err.status == 404:
            bank_exists = False
        else:
            raise
    except Exception as e:
        err_lower = str(e).lower()
        if "404" in err_lower or "not found" in err_lower:
            bank_exists = False
        elif "409" in err_lower or "already exists" in err_lower:
            bank_exists = True
        else:
            raise

    if bank_exists:
        print(f"Hindsight bank already exists: {bank_id}")
    else:
        try:
            client.create_bank(bank_id=bank_id)
            print(f"Hindsight bank created successfully: {bank_id}")
        except Exception as create_err:
            err_lower = str(create_err).lower()
            if "409" in err_lower or "already exists" in err_lower:
                print(f"Hindsight bank already exists: {bank_id}")
            else:
                raise create_err

except Exception as e:
    err_str = sanitize_msg(str(e))
    print(f"Error setting up Hindsight bank '{bank_id}': {err_str}")
    sys.exit(1)

# Memory retention test section
memory_content = "Customer Priya Nair from Coastal Foods Ltd prefers a concise ROI-focused discussion and previously raised concerns about implementation cost."

try:
    client.retain(
        bank_id=bank_id,
        content=memory_content
    )
    print("Memory retained successfully")
except Exception as e:
    err_str = sanitize_msg(str(e))
    print(f"Error retaining memory: {err_str}")
    sys.exit(1)

# Memory recall test section
recall_query = "What does Priya Nair from Coastal Foods Ltd care about in a sales discussion?"

try:
    recall_response = client.recall(
        bank_id=bank_id,
        query=recall_query
    )
    print(f"\nRecall results for query: \"{recall_query}\"")
    if hasattr(recall_response, "results") and recall_response.results:
        for idx, res in enumerate(recall_response.results, 1):
            text = getattr(res, "text", str(res))
            res_type = getattr(res, "type", "fact")
            print(f"  {idx}. [{res_type}] {text}")
    else:
        print("  No memory results returned.")
except Exception as e:
    err_str = sanitize_msg(str(e))
    print(f"Error recalling memory: {err_str}")
    sys.exit(1)
finally:
    try:
        client.close()
    except Exception:
        pass
