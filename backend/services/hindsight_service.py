import os
from pathlib import Path

from dotenv import load_dotenv
try:
    from hindsight_client import Hindsight
except ImportError:
    Hindsight = None


# Load .env from the project root or backend directory
PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASE_DIR = Path(__file__).resolve().parents[1]

for env_file in [BASE_DIR / ".env", PROJECT_ROOT / ".env"]:
    if env_file.exists():
        load_dotenv(env_file)

api_key = os.getenv("HINDSIGHT_API_KEY")

base_url = os.getenv(
    "HINDSIGHT_BASE_URL",
    "https://api.hindsight.vectorize.io"
)

bank_id = os.getenv(
    "HINDSIGHT_BANK_ID",
    "dealsight-sales"
)

client = None

if api_key and Hindsight is not None:
    try:
        client = Hindsight(
            base_url=base_url,
            api_key=api_key
        )
    except Exception as e:
        print(f"[HINDSIGHT WARNING] Failed to initialize client: {e}")
        client = None


def retain_memory(content: str, tags: list[str] | None = None):
    """Store a memory in Hindsight."""
    if client is None:
        return None

    try:
        return client.retain(
            bank_id=bank_id,
            content=content,
            tags=tags
        )
    except Exception:
        return None


def recall_memory(
    query: str,
    tags: list[str] | None = None
):
    """Retrieve relevant memories from Hindsight."""
    if client is None:
        return []

    try:
        response = client.recall(
            bank_id=bank_id,
            query=query,
            tags=tags,
            tags_match="all_strict"
        )

        if hasattr(response, "results"):
            return response.results

        return []

    except Exception:
        return []
def close_hindsight():
    """Close the Hindsight client."""
    if client is not None:
        client.close()