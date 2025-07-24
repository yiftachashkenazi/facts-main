import json
import logging
import os
from typing import Dict, List

from google import genai
from google.genai.types import GenerateContentConfig

logger = logging.getLogger(__name__)

SYSTEM_INSTRUCTION = """You are a narrative event extraction assistant. Your task is to analyze the INPUT TEXT and extract a sequence of events that form a coherent narrative.

For each event you identify, return a JSON object with the following structure:

{
    "event_numeric_id": 1,
    "is_first_event": true/false,
    "linkage_to_previous_event_in_text": "none" or "temporal" or "causal",
    "event": "Summary of the event",
    "characters": ["Character A", "Character B"],
    "class": "narrative" or "quote" or "opening" or "conclusion",
    "quote": "Verbatim quote if any",
    "direct_quote": true/false,
    "reference_to_quote_or_conclusion": "narrator" or speaker's name,
    "time_frame": "past" or "present" or "future",
    "estimate_time": "Timestamp or contextual inference",
    "details": "Additional details about the event",
    "is_last_event": true/false
}

Return an array of these event objects in chronological order."""

# --- LLM Event Extraction Function ---
def generate_events_for_text(text: str) -> str:
    """
    Send <text> to Gemini‑Flash with the narrative‑event SYSTEM_INSTRUCTION
    and return the raw plain‑text response (expected JSON).
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        logger.error("GEMINI_API_KEY environment variable not set")
        raise ValueError("GEMINI_API_KEY environment variable not set")

    # Create client with the new Google Gen AI SDK
    client = genai.Client(api_key=api_key)

    prompt = f"{SYSTEM_INSTRUCTION}\n\nINPUT TEXT:\n{text}"
    
    # Define the API call function
    def make_api_call():
        return client.models.generate_content(
            model='gemini-2.0-flash',
            contents=prompt,
            config=GenerateContentConfig(
                temperature=0.2,
                max_output_tokens=15000,  # Set to 15k to handle complex event extraction
                # No safety settings needed as we're using newer format with BLOCK_NONE default
            )
        )

    response = make_api_call()
    return response.text.strip() 