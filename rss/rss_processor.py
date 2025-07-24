#!/usr/bin/env python3
# File: rss/rss_processor.py
"""
RSS processing module using sentence transformers and Gemini for event matching.
- Computes similarity between user input and RSS articles.
- Uses Gemini to identify matching events.
- Extracts structured events for TRUE matches and saves to rss_events.sql.
- Returns structured event data for the main pipeline.
"""

import asyncio
import json
import logging
import os
import re
import time
from datetime import datetime, timedelta, timezone
from hashlib import md5
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import aiohttp
import feedparser
import requests
import schedule
import wikipedia
import yaml
from bs4 import BeautifulSoup
from dateutil import parser
from dotenv import load_dotenv
from google import genai

from simple_similarity import SentenceTransformer, cosine_similarity
from utils.json_utils import parse_json_any

# ——————————————————————————————————————————— #
# Configure Logging
# ——————————————————————————————————————————— #
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('debug.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ——————————————————————————————————————————— #
# Config
# ——————————————————————————————————————————— #
JSON_PATH = "data/rss_articles.json"
VECTORS_DIR = "data/vectors"
MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"  # Supports Hebrew
EVENTS_SQL_FILENAME = "rss_events.sql"

# ——————————————————————————————————————————— #
# Narrative Event Extraction System Instruction
# ——————————————————————————————————————————— #
SYSTEM_INSTRUCTION = r"""
role: system
content: |
  You are a narrative analysis expert, specializing in event detection, quotation, and structural annotation.
  Your task is to read a given text and extract key events into structured JSON objects according to the schema below.
  Always quote directly from the text for each relevant field (in the text's original language), justify your decision at each step, and use a decision tree logic for your reasoning.

  INSTRUCTIONS:
  1. Read the entire text carefully.
  2. Detect and order events by their actual timeline, not by their appearance in the text.
  3. For each event, fill in the following JSON fields:
     - The "event" field must summarize the event in **6 words or fewer**. Word choice is critical: use the most informative, efficient phrasing to capture the core action or claim.
     - If using a "narrative" field, it must also be **6 words or fewer**.
     - The "quote" field must contain the **entire, unedited quote** from the text, as it appears—do not summarize, shorten, or omit any part of the quoted segment.
     - If a single sentence contains both **direct and indirect quotes**, treat each as a **separate event** with its own JSON object and its corresponding full quote.
     - Always quote the exact words from the text for any quote field.
     - If the text is in another language, use the original for names and quotes.
     - For "estimate_time", you must always provide a value. If no explicit date or time is given, analyze the sentence for context clues and make your best estimate or inference. Never leave this field empty. If truly uncertain, provide an inferred answer such as "unspecified, but past", "unknown, likely recent", or "implied: future".

  4. Follow the decision tree below for each field.

  JSON OUTPUT SCHEMA:
    event_numeric_id: 1
    is_first_event: true
    linkage_to_primer_event_in_text: "none"
    event: "<summarize the event in 6 words or fewer>"
    characters:
      - "list all main characters, as named in the text"
    class: "narrative/quote/opening/conclusion"
    quote: "entire, unedited quote as appears in the text"
    direct_quote: true
    reference_to_quote_or_conclusion: "X/narrator/null if narrative"
    time_frame: "past/present/future"
    estimate_time: "estimate or explicit date if present, or inferred based on context (never empty)"
    details: "numbers, data, context"
    is_last_event: false

  DECISION TREE LOGIC:
    Step 1:
      - Does the sentence/segment describe a clear action or state?
      - If YES: Proceed.
      - If NO: Skip or continue scanning.
    Step 2:
      - Is this the first event in the timeline?
      - If YES: Set "is_first_event": true.
      - If NO:  Set "is_first_event": false.
    Step 3:
      - What is its linkage to previous events (if any)?
      - "none" if unrelated,
      - "reaction" if responding to the previous event,
      - "pivot" if it changes the story's direction,
      - "analogy" if comparing,
      - "happened later with no reason",
      - "context giving" if providing background.
    Step 4:
      - Event: Summarize the core action or claim in 6 words or fewer.
    Step 5:
      - Characters: List all named entities involved (use their original text language).
    Step 6:
      - Class: Categorize as "narrative", "quote", "opening", or "conclusion".
    Step 7:
      - Quote: Include the entire, unedited quote relevant to that event.
    Step 8:
      - Direct quote: true if quoted verbatim, false if paraphrased.
    Step 9:
      - Reference to quote or conclusion: Who is making the statement?
    Step 10:
      - Time frame: Is the event described in past, present, or future tense?
    Step 11:
      - Estimate time: Explicit or inferred; never empty.
    Step 12:
      - Details: Numbers, data, or context if present.
    Step 13:
      - Is last event: true if this is the final chronological event.

  Final Instructions:
    - Always order events by timeline, even if narrative order differs.
    - If uncertain, explain reasoning in a comment before the JSON output.
    - Each JSON event must be self‑contained.
"""

# ——————————————————————————————————— #
# Helpers
# ——————————————————————————————————— #
def load_articles(path: str) -> List[Dict]:
    """Load RSS articles from JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def vector_paths(aid: str, vectors_dir: str) -> tuple:
    """Return paths for title and text vectors."""
    return (
        os.path.join(vectors_dir, f"{aid}_title.npy"),
        os.path.join(vectors_dir, f"{aid}_text.npy"),
    )

def load_vector(path: str) -> List[float]:
    """Load a vector from file (now using simple lists instead of numpy)."""
    try:
        # Try to load as numpy first for backward compatibility
        import numpy as np
        vec = np.load(path)
        return vec.tolist()
    except Exception:
        # If numpy not available or file format different, return empty vector of correct expected dimension
        # Assuming sentence transformer model outputs 768 dim vectors, adjust if different
        # For "paraphrase-multilingual-MiniLM-L12-v2", it's 384
        logger.warning(f"Failed to load vector from {path}, returning zero vector.")
        return [0.0] * 384 # Default to 384 dimensions for MiniLM

def cosine(u: List[float], v: List[float]) -> float:
    """Compute cosine similarity between two vectors."""
    # Add a check for empty or zero vectors to prevent division by zero if not handled by cosine_similarity
    if not u or not v or all(x == 0 for x in u) or all(x == 0 for x in v):
        return 0.0
    return cosine_similarity(u, v)

def time_adjust(pub_str: str) -> float:
    """Adjust similarity score based on publication recency."""
    try:
        pub = parser.parse(pub_str).astimezone(timezone.utc)
        delta = datetime.now(timezone.utc) - pub
        return 0.03 if delta.days <= 7 else -0.03
    except:
        return 0.0

def build_json_result(article: Dict, score: float) -> Dict:
    """Build JSON result with similarity score."""
    result = article.copy()
    result["similarity"] = round(float(score), 4)
    return result

def generate_event_matches(user_input: str, rss_feed_entries: List[Dict]) -> Dict:
    """
    Use Gemini to determine which RSS items describe the same event.
    Returns dict: entry_id -> {"Match": bool, "Quote": str}.
    """
    logger.debug("Starting generate_event_matches")
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        logger.error("GEMINI_API_KEY environment variable not set")
        raise ValueError("GEMINI_API_KEY environment variable not set")

    client = genai.Client(api_key=api_key)

    prompt_lines = [
        "You are a data engineer tasked with correlating news items.",
        "For each entry in the JSON list called rss_feed_entries decide whether it reports the *same real‑world event* as user_input.",
        "Output *ONLY* one pipe‑delimited line per entry with the fields:",
        "entry_id | Match | Quote",
        "Use TRUE or FALSE (uppercase) for Match.",
        "Quote should be ≤20 words taken verbatim from the entry that justifies the match.",
        "If Match is FALSE leave Quote blank but still include the trailing pipe.",
        "Do not output anything else—no headers, explanations, or code fences."
    ]
    prompt = (
        "\n".join(prompt_lines)
        + "\n\nuser_input:\n"
        + user_input
        + "\n\nrss_feed_entries:\n"
        + json.dumps(rss_feed_entries, ensure_ascii=False)
    )

    logger.debug("Prompt prepared for Gemini")
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt,
        config={"temperature": 0.2}
    )
    response_text = response.text.strip() if response.text else ""
    logger.debug("Gemini raw response: %s", response_text)

    matches = {}
    for line in response_text.splitlines():
        parts = [p.strip() for p in line.split("|", 2)]
        if len(parts) >= 2:
            entry_id = parts[0]
            is_match = parts[1].upper() == "TRUE"
            quote = parts[2] if len(parts) == 3 else ""
            matches[entry_id] = {"Match": is_match, "Quote": quote}
    logger.debug("Parsed match dict: %s", matches)
    return matches

def generate_events_for_text(text: str) -> str:
    """
    Send text to Gemini-Flash for narrative event extraction.
    Returns raw plain-text JSON response.
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        logger.error("GEMINI_API_KEY environment variable not set")
        raise ValueError("GEMINI_API_KEY environment variable not set")

    client = genai.Client(api_key=api_key)

    prompt = f"{SYSTEM_INSTRUCTION}\n\nINPUT TEXT:\n{text}"
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt,
        config={"temperature": 0.2}
    )
    return response.text.strip() if response.text else ""

# ——————————————————————————————————— #
# Main RSS Processing Function
# ——————————————————————————————————— #
def process_rss_with_gemini(
    user_text: str,
    json_path: str = JSON_PATH,
    vectors_dir: str = VECTORS_DIR,
    sql_output: str = EVENTS_SQL_FILENAME
) -> List[Dict]:
    """
    Process RSS articles to find matches and extract structured events.
    - Computes similarity using sentence transformers.
    - Uses Gemini to identify matching events.
    - Saves structured events to SQL file for TRUE matches.
    - Returns list of matched events for the main pipeline.
    """
    load_dotenv()

    # Validate input
    if not user_text or not user_text.strip():
        logger.warning("Empty or invalid user_text provided")
        return []

    # Initialize model
    try:
        model = SentenceTransformer(MODEL_NAME)
        logger.debug(f"Model loaded: {MODEL_NAME}")
    except Exception as exc:
        logger.error(f"Failed to load model {MODEL_NAME}: {exc}")
        return []

    # Encode user text
    try:
        q_vec_np = model.encode(user_text)
        q_vec: List[float] = q_vec_np.tolist() # Convert to list
        logger.debug(f"User text encoded, embedding shape: {q_vec_np.shape}")
    except Exception as exc:
        logger.error(f"Failed to encode user_text: {exc}")
        return []

    # Load articles
    try:
        articles = load_articles(json_path)
        logger.debug(f"Loaded {len(articles)} articles from {json_path}")
    except Exception as exc:
        logger.error(f"Failed to load articles from {json_path}: {exc}")
        return []

    # Collect similarities
    scored_articles: Dict[str, Dict[str, Any]] = {}
    for art in articles:
        aid = md5(art["link"].encode("utf-8")).hexdigest()
        t_path, x_path = vector_paths(aid, vectors_dir)
        if not (os.path.isfile(t_path) and os.path.isfile(x_path)):
            logger.debug(f"Vector files missing for article ID {aid}")
            continue

        try:
            v_title = load_vector(t_path)
            v_text = load_vector(x_path)
        except Exception as exc:
            logger.debug(f"Failed to load vectors for article ID {aid}: {exc}")
            continue

        title_sim = cosine(q_vec, v_title)
        content_sim = cosine(q_vec, v_text)
        best_sim = max(title_sim, content_sim)
        adj = time_adjust(art.get("published", ""))
        score = best_sim + adj

        try:
            pub_date_str = art.get("published", "")
            if pub_date_str and pub_date_str != "None":
                pub_date = parser.parse(pub_date_str).astimezone(timezone.utc)
            else:
                pub_date = datetime.min.replace(tzinfo=timezone.utc) # Default for missing date
        except Exception as e:
            logger.warning(f"Could not parse date string '{pub_date_str}': {e}")
            pub_date = datetime.min.replace(tzinfo=timezone.utc)

        link = art["link"]
        # Ensure score is float before comparison
        current_score = scored_articles.get(link, {}).get("score", float("-inf"))
        if score > current_score:
            scored_articles[link] = {
                "score": float(score),
                "sim": float(best_sim),
                "pub_date": pub_date,
                "article": art
            }

    # Sort and select top results
    sorted_results = sorted(scored_articles.values(), key=lambda x: float(x["score"]), reverse=True)
    top_results = sorted_results[:10]
    logger.debug(f"Selected {len(top_results)} top articles")

    # Create JSON output for processing
    output: List[Dict[str, Any]] = [build_json_result(item["article"], float(item["score"])) for item in top_results]

    # Prepare payload for event matching
    rss_payload = [
        {
            "entry_id": item.get("link", ""),
            "text": item.get("summary", ""),
            "other_metadata": item
        }
        for item in output
    ]

    # Use LLM to classify matches
    try:
        matches_dict = generate_event_matches(user_text, rss_payload)
        logger.debug(f"Generated matches for {len(matches_dict)} entries")
    except Exception as exc:
        logger.error(f"Event matching failed: {exc}")
        return []

    # Generate structured events for TRUE matches and write to SQL
    event_count = 0
    sql_path = Path(sql_output).resolve()
    payload_by_id = {entry["entry_id"]: entry for entry in rss_payload}
    rss_feed_results = []

    with open(sql_path, "w", encoding="utf-8") as f:
        f.write("\nCREATE TABLE IF NOT EXISTS rss_events (\n"
                "    entry_id VARCHAR(255) PRIMARY KEY,\n"
                "    events_json TEXT\n"
                ");\n\n")

        for entry_id, info in matches_dict.items():
            if not info.get("Match"):
                continue  # Skip FALSE matches

            entry = payload_by_id.get(entry_id)
            if not entry:
                logger.warning("Gemini returned unknown entry_id: %s", entry_id)
                continue

            meta = entry["other_metadata"]
            title = meta.get("title", "")
            summary = meta.get("summary", "")
            full_text = meta.get("content") or meta.get("text") or meta.get("summary", "")
            published = meta.get("published", "")

            # Split into paragraphs
            paragraphs = re.split(r"\n\s*\n", full_text.strip()) if full_text else [""]
            if len(paragraphs) <= 2:
                paragraphs = ["\n".join(paragraphs)]

            for idx, para in enumerate(paragraphs, 1):
                if not para.strip():
                    continue
                para_id = f"{entry_id}#p{idx}" if len(paragraphs) > 1 else entry_id

                user_text_para = (
                    f"Title: {title}\n"
                    f"Summary: {summary}\n"
                    f"Full text paragraph {idx}/{len(paragraphs)}:\n{para}\n"
                    f"Published: {published}\n"
                    f"Link: {entry_id}"
                )

                try:
                    events_json_raw = generate_events_for_text(user_text_para).strip()
                    # Remove code fences if present
                    if events_json_raw.startswith("```"):
                        events_json_raw = re.sub(r"^```[a-zA-Z0-9_-]*\n", "", events_json_raw, count=1, flags=re.MULTILINE)
                        if events_json_raw.endswith("```"):
                            events_json_raw = events_json_raw[:-3].rstrip()

                    events_json_sql = events_json_raw.replace("'", "''")
                    # Parse JSON for inclusion in rss_feed_results
                    try:
                        events_data = json.loads(events_json_raw)
                    except json.JSONDecodeError as e:
                        logger.error("Failed to parse events JSON for %s: %s", para_id, e)
                        continue

                    # Add to results
                    rss_feed_results.append({
                        "entry_id": para_id,
                        "title": title,
                        "summary": summary,
                        "published": published,
                        "link": entry_id,
                        "events": events_data,
                        "similarity": meta.get("similarity", 0.0),
                        "quote": info.get("Quote", "")
                    })

                    # Write to SQL
                    f.write("INSERT INTO rss_events (entry_id, events_json) VALUES (\n"
                            f"    '{para_id}',\n"
                            f"    '{events_json_sql}'\n"
                            ");\n\n")
                    event_count += 1
                except Exception as exc:
                    logger.error("Event extraction failed for %s: %s", para_id, exc)
                    continue

    print(f"✔  Events SQL written to {sql_path}")
    print(f"Processed {event_count} individual events for LLM.")
    return rss_feed_results

async def extract_author_from_text(text: str) -> str:
    """
    Extract author name from text using Gemini.
    
    Args:
        text (str): The input text
        
    Returns:
        str: Extracted author name or "Unknown"
    """
    logger.info("🔍 Extracting author from text using Gemini...")
    
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set")

    client = genai.Client(api_key=api_key)

    prompt = f"{SYSTEM_INSTRUCTION}\n\nINPUT TEXT:\n{text}"
    
    try:
        response = client.models.generate_content(
            model='gemini-2.0-flash',
            contents=prompt
        )
        
        response_text = response.text if response.text else ""
        if response_text:
            result = parse_json_any(response_text)
            author_name = result.get("author_name", "Unknown")
            logger.info(f"✅ Author extracted: {author_name}")
            return author_name
        else:
            logger.warning("Empty response from Gemini")
            return "Unknown"
        
    except Exception as e:
        logger.error(f"❌ Error extracting author: {e}")
        return "Unknown"

# --- RSS Feed Processing Function ---
def process_rss_feed(feed_url: str) -> List[Dict[str, Any]]:
    """
    Process an RSS feed and return a list of articles with their metadata.
    """
    try:
        feed = feedparser.parse(feed_url)
        articles = []
        
        for entry in feed.entries:
            article = {
                "title": entry.title,
                "link": entry.link,
                "published": entry.get("published", ""),
                "summary": entry.get("summary", ""),
                "content": entry.get("content", [{"value": ""}])[0].get("value", ""),
                "processed_date": datetime.now().isoformat()
            }
            articles.append(article)
            
        return articles
    except Exception as e:
        logger.error(f"Error processing RSS feed {feed_url}: {str(e)}")
        return []

if __name__ == "__main__":
    # Simple test
    sample_text = (
        "ישראל שחררה את האסירים אתמול מעזה.\n"
        "לדעתי זה היה צעד בלתי-חוקי."
    )
    results = process_rss_with_gemini(
        user_text=sample_text,
        json_path=JSON_PATH,
        vectors_dir=VECTORS_DIR,
        sql_output="rss_events_test.sql"
    )
    print(f"Returned {len(results)} RSS event results.")