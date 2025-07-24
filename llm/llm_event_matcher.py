"""
gemini_event_matcher.py   – robust version
"""
import json
import logging
import os
import re
from typing import Dict, List, Optional

from google import genai
from google.genai.types import GenerateContentConfig

from utils.json_utils import extract_json_block, parse_json_any  # unchanged

logger = logging.getLogger(__name__)

SYSTEM_INSTRUCTION = r"""
You are a fact-checking assistant. Your task is to analyse whether the events
described in the ARTICLE TEXT match or relate to the EVENTS provided.

ENHANCED ANALYSIS REQUIREMENTS:
- Check provided Wiki data FIRST for background context and historical information
- For abstract or complex events, Wiki often contains essential context
- Use RSS data for recent news and current developments
- Generate specific search queries for each event when needed
- Provide detailed source information for user verification

For each event in EVENTS, determine if there is a matching or related event in
the ARTICLE TEXT.  Return a *single* JSON object with this exact schema:

{
  "matches": [
    {
      "event_id": "<ID from EVENTS>",
      "is_match": true | false,
      "confidence": 0.0-1.0,
      "explanation": "<detailed analysis 15-25 words>",
      "quote": "<relevant quotation if matched, else empty string>",
      "arguments_extracted": ["argument 1", "argument 2"],
      "sources_used": [
        {
          "type": "Wiki/RSS/Search",
          "source_name": "Wikipedia/BBC News/Reuters/etc",
          "title": "Page or article title", 
          "url": "exact_url",
          "snippet": "relevant_excerpt",
          "date_published": "YYYY-MM-DD or original format",
          "relevance_score": 0.0
        }
      ],
      "search_queries_generated": ["query1", "query2"]
    }
  ]
}

ANALYSIS PROCESS:
1. Extract individual arguments from each event description
2. Check Wiki data for background context
3. Check RSS data for recent developments  
4. Generate specific search queries if needed
5. Provide detailed explanation with source verification

Think step-by-step but **output only the JSON**.  
Be strict: mark `"is_match": true` only when you are confident the two events
are the same or directly related.
"""

# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #

def _safe_json_loads(text: str) -> dict | None:
    """Try to load a JSON string, applying a couple of common fixes first."""
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # common Gemini escape patterns
        text = text.replace("\\\\", "\\").replace('\\"', '"').strip("` \n")
        try:
            return json.loads(text)
        except json.JSONDecodeError as err:
            logger.error("Still cannot parse JSON after fixes: %s", err)
            return None


def _extract_matches(obj: dict) -> Dict[str, Dict[str, str | bool]]:
    """
    Convert the assistant's JSON response into the flattened
    {event_id: {"Match": bool, "Quote": str}} shape expected by main.py
    """
    results: Dict[str, Dict[str, str | bool]] = {}
    for item in obj.get("matches", []):
        eid = item.get("event_id")
        if eid:
            results[eid] = {
                "Match":  bool(item.get("is_match", False)),
                "Quote":  item.get("quote", "").strip()
            }
    return results


# --------------------------------------------------------------------------- #
# Main callable                                                                #
# --------------------------------------------------------------------------- #

def generate_event_matches(events: List[dict], article_text: str) -> Dict[str, dict]:
    """
    Compare EVENTS (list-of-dicts) with ARTICLE TEXT using Gemini Flash and
    return a flat dictionary keyed by event_id.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError("GEMINI_API_KEY environment variable not set")

    # Create client with the new Google Gen AI SDK
    client = genai.Client(api_key=api_key)

    prompt = (
        SYSTEM_INSTRUCTION
        + "\n\nEVENTS:\n"
        + json.dumps(events, ensure_ascii=False, indent=2)
        + "\n\nARTICLE TEXT:\n"
        + article_text
    )

    try:
        # Define the API call function 
        def make_api_call():
            return client.models.generate_content(
                model='gemini-2.0-flash',
                contents=prompt,
                config=GenerateContentConfig(
                    temperature=0.2,
                    max_output_tokens=10000,
                    # No safety settings needed as we're using newer format with BLOCK_NONE default
                )
            )

        response = make_api_call()
        assistant_reply = response.text.strip()
    except Exception as exc:
        logger.error("Gemini request failed: %s", exc, exc_info=True)
        return {}

    # --- Pull the JSON block from the assistant reply --------------------- #
    json_fragment = (
        extract_json_block(assistant_reply)        # your utility – returns '' if none
        or parse_json_any(assistant_reply)         # fall-back – tries to find *any* JSON
    )

    parsed = _safe_json_loads(json_fragment)
    if parsed is None:
        logger.error("No valid JSON found in Gemini output.\nRaw output:\n%s", assistant_reply)
        return {}

    return _extract_matches(parsed)