import os
import json
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
client = Groq(api_key=groq_api_key) if groq_api_key else None
MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")


def analyze_conversation(conversation: str):
    if client is None:
        raise ValueError("GROQ_API_KEY is missing. Add it to your .env file.")
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": """
                You are a Deal Intelligence Agent.
                Analyze sales conversations and return valid JSON with:
                summary, objections, competitors, pain_points,
                priorities, buying_signals, risks, action_items.
                Do not invent facts. Separate customer statements
                from assumptions.
                """
            },
            {
                "role": "user",
                "content": conversation
            }
        ],
        response_format={"type": "json_object"},
        temperature=0.2
    )

    content = response.choices[0].message.content
    return json.loads(content)