#!/usr/bin/env python3

import asyncio
import json
import logging
import os
import re
from datetime import datetime, timezone
from hashlib import md5
from pathlib import Path
from typing import Optional

import schedule
import yaml
from dateutil import parser
from dotenv import load_dotenv
from google import genai

from llm.llm_event_correlator import generate_event_correlation_single
from llm.llm_event_extractor import generate_events_for_text
from llm.llm_fact_checker import generate_fact_check
from rss.rss_collector import build_rss_feeds_collection
from rss.rss_processor import (
    build_json_result,
    cosine,
    load_articles,
    load_vector,
    time_adjust,
    vector_paths,
)
from simple_similarity import SentenceTransformer
from utils.json_utils import extract_json_block, parse_json_any
from wiki.wiki_processor import (
    get_author_info,
    get_author_info_by_name,
    process_characters_wiki,
)

# --- Configure Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- CONFIG ---
JSON_PATH = "data/rss_articles.json"
VECTORS_DIR = "data/vectors"
MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

def generate_event_matches(user_input, rss_feed_entries):
    """Fast event matching using Gemini Flash."""
    logger.info("Starting event matching")
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set")

    # genai.configure(api_key=api_key) # Not needed if GOOGLE_API_KEY or GEMINI_API_KEY is set
    model = genai.GenerativeModel("gemini-2.0-flash", api_key=api_key)

    prompt = (
        "For each RSS entry, decide if it reports the same event as user_input.\n"
        "Output format: entry_id | TRUE/FALSE | quote\n"
        f"user_input: {user_input}\n"
        f"rss_entries: {json.dumps(rss_feed_entries, ensure_ascii=False)}"
    )

    response = model.generate_content(prompt, generation_config={"temperature": 0.1})
    
    matches = {}
    for line in response.text.strip().splitlines():
        parts = [p.strip() for p in line.split("|", 2)]
        if len(parts) >= 2:
            entry_id = parts[0]
            is_match = parts[1].upper() == "TRUE"
            quote = parts[2] if len(parts) == 3 else ""
            matches[entry_id] = {"Match": is_match, "Quote": quote}
    
    return matches

async def process_rss_data(query: str, model: SentenceTransformer) -> tuple:
    """Process RSS data concurrently."""
    logger.info("🔍 DEBUG: 📡 Processing RSS data")
    logger.info(f"🔍 DEBUG: 📡 Query for RSS: {query[:100]}...")

    q_vec = model.encode(query)
    logger.info(f"🔍 DEBUG: 📡 Query encoded to vector of length: {len(q_vec)}")
    
    articles = load_articles(JSON_PATH)
    logger.info(f"🔍 DEBUG: 📡 Loaded {len(articles)} articles from {JSON_PATH}")

    # Pre-filter articles that have vector files
    articles_with_vectors = []
    for art in articles:
        if isinstance(art, dict) and "link" in art:
            aid = md5(art["link"].encode("utf-8")).hexdigest()
            t_path, x_path = vector_paths(aid, VECTORS_DIR)
            if os.path.isfile(t_path) and os.path.isfile(x_path):
                articles_with_vectors.append(art)
    
    logger.info(f"🔍 DEBUG: 📡 Found {len(articles_with_vectors)} articles with vector files out of {len(articles)}")
    
    if not articles_with_vectors:
        logger.warning("🔍 DEBUG: 📡 No articles with vector files found!")
        return [], {}
    
    # Process articles concurrently - use available articles
    tasks = []
    articles_to_process = articles_with_vectors[:min(30, len(articles_with_vectors))]
    for art in articles_to_process:
        tasks.append(process_single_article(art, q_vec))
    
    logger.info(f"🔍 DEBUG: 📡 Created {len(tasks)} article processing tasks")
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Filter successful results
    scored_articles = {}
    successful_results = 0
    for result in results:
        if isinstance(result, dict) and result:  # Ensure non-empty dict
            scored_articles.update(result)
            successful_results += 1
    
    logger.info(f"🔍 DEBUG: 📡 Successfully processed {successful_results}/{len(tasks)} articles")
    
    # Get top 8 results instead of 10 for faster processing
    sorted_results = sorted(scored_articles.values(), key=lambda x: x["score"], reverse=True)[:8]
    logger.info(f"🔍 DEBUG: 📡 Selected top {len(sorted_results)} articles by similarity score")
    
    if sorted_results:
        logger.info(f"🔍 DEBUG: 📡 Best similarity score: {sorted_results[0]['score']:.4f}")
        logger.info(f"🔍 DEBUG: 📡 Worst similarity score: {sorted_results[-1]['score']:.4f}")
    
    # Build RSS payload
    rss_payload = []
    for item in sorted_results:
        try:
            result = build_json_result(item["article"], item["score"])
            if result and "link" in result and "summary" in result:
                rss_payload.append({
                    "entry_id": result["link"],
                    "text": result["summary"],
                    "other_metadata": result
                })
        except Exception as e:
            logger.error(f"🔍 DEBUG: 📡 Failed to build RSS result: {e}")
            continue
    
    logger.info(f"🔍 DEBUG: 📡 Built RSS payload with {len(rss_payload)} articles")
    
    # Generate event matches
    if rss_payload:
        logger.info("🔍 DEBUG: 📡 Generating event matches with Gemini...")
        matches_dict = generate_event_matches(query, rss_payload)
        matches_count = len([m for m in matches_dict.values() if m.get("Match")])
        logger.info(f"🔍 DEBUG: 📡 Event matching complete - {matches_count}/{len(matches_dict)} articles matched")
    else:
        logger.warning("🔍 DEBUG: 📡 No RSS payload available for event matching")
        matches_dict = {}
    
    return rss_payload, matches_dict

async def process_single_article(art: dict, q_vec) -> dict:
    """Process a single article for similarity scoring."""
    try:
        aid = md5(art["link"].encode("utf-8")).hexdigest()
        t_path, x_path = vector_paths(aid, VECTORS_DIR)
        
        # Load vectors (we know they exist from pre-filtering)
        v_title = load_vector(t_path)
        v_text = load_vector(x_path)
        
        # Calculate similarity scores
        title_sim = cosine(q_vec, v_title)
        content_sim = cosine(q_vec, v_text)
        best_sim = max(title_sim, content_sim)
        adj = time_adjust(art.get("published", ""))
        score = best_sim + adj
        
        logger.info(f"🔍 DEBUG: 📄 Article {art['link'][:30]}... - Score: {score:.4f} (sim: {best_sim:.4f})")
        
        try:
            pub_date = parser.parse(art.get("published", "")).astimezone(timezone.utc)
        except:
            pub_date = datetime.min.replace(tzinfo=timezone.utc)
        
        link = art["link"]
        return {link: {
            "score": score,
            "sim": best_sim,
            "pub_date": pub_date,
            "article": art
        }}
    except Exception as e:
        logger.error(f"🔍 DEBUG: 📄 Failed to process article: {e}")
        return {}

async def process_user_events(query: str) -> tuple:
    """Process user events extraction."""
    logger.info("🔍 DEBUG: 📝 Processing user events")
    logger.info(f"🔍 DEBUG: 📝 Query for events: {query[:100]}...")
    
    try:
        user_events_json_raw = generate_events_for_text(query).strip()
        logger.info(f"🔍 DEBUG: 📝 Raw events JSON length: {len(user_events_json_raw)}")
        
        user_events_json_raw = extract_json_block(user_events_json_raw)
        user_events = parse_json_any(user_events_json_raw)
        
        if not isinstance(user_events, list):
            user_events = [user_events] if user_events else []
            
        logger.info(f"🔍 DEBUG: 📝 Successfully parsed {len(user_events)} events")
        
        # Log sample events
        for i, event in enumerate(user_events[:3]):  # Log first 3 events
            event_text = event.get("event", "No event text")
            logger.info(f"🔍 DEBUG: 📝 Event {i+1}: {event_text[:50]}...")
            
    except Exception as e:
        logger.error(f"🔍 DEBUG: 📝 Failed to process user events: {e}")
        user_events = []
        user_events_json_raw = "[]"
    
    return user_events, user_events_json_raw

async def process_wikipedia_data(user_events_json_raw: str, query: str, author_name: Optional[str] = None) -> tuple[dict, dict]:
    """Process Wikipedia data extraction."""
    logger.info("🔍 DEBUG: 📚 Processing Wikipedia data")
    logger.info(f"🔍 DEBUG: 📚 Author name passed to Wikipedia processing: {author_name}")
    
    try:
        # Process both character info and author info concurrently
        char_task = process_characters_wiki(user_events_json_raw, query)
        
        # Pass the author name directly if available
        if author_name:
            logger.info(f"🔍 DEBUG: 📚 Using provided author name: {author_name}")
            author_task = get_author_info_by_name(author_name)
        else:
            logger.info("🔍 DEBUG: 📚 Extracting author from text")
            author_task = get_author_info(query)
        
        logger.info("🔍 DEBUG: 📚 Starting concurrent Wikipedia processing...")
        results = await asyncio.gather(char_task, author_task, return_exceptions=True)
        
        char_info_res = results[0]
        author_info_res = results[1]

        # Process character info results
        if isinstance(char_info_res, dict):
            char_info = char_info_res
            logger.info(f"🔍 DEBUG: 📚 Character extraction complete - Found {len(char_info)} characters")
            
            # Log sample characters
            if char_info:
                sample_chars = list(char_info.keys())[:3]
                for char in sample_chars:
                    logger.info(f"🔍 DEBUG: 📚 Character found: {char}")
        elif isinstance(char_info_res, Exception):
            logger.error(f"🔍 DEBUG: 📚 Failed to process character info: {char_info_res}")
            char_info = {}

        # Process author info results
        if isinstance(author_info_res, dict):
            author_info = author_info_res
            if author_info and author_info.get("name"):
                logger.info(f"🔍 DEBUG: 📚 Author info extracted: {author_info['name']}")
            else:
                logger.warning("🔍 DEBUG: 📚 No author info found in results")
        elif isinstance(author_info_res, Exception):
            logger.error(f"🔍 DEBUG: 📚 Failed to process author info: {author_info_res}")
            author_info = {}
    
    except Exception as e:
        logger.error(f"🔍 DEBUG: 📚 Failed to process Wikipedia data: {e}")
        char_info = {}
        author_info = {}
    
    return char_info, author_info

async def generate_calibration_data(user_events: list, rss_payload: list) -> list:
    """Generate calibration data for events."""
    logger.info("Generating calibration data")
    
    if not user_events or not rss_payload:
        return []
    
    # Limit processing for speed - take only top events
    limited_user_events = user_events[:5]  # Process max 5 events for speed
    limited_rss_payload = rss_payload[:6]  # Process max 6 RSS items for speed
    
    # Build RSS feeds collection (as a list of dicts, one for each source)
    rss_feeds_collection_list = []
    
    # Concurrently generate events for each RSS item
    rss_event_tasks = []
    for item in limited_rss_payload:
        rss_event_tasks.append(generate_events_for_single_rss_item(item))
    
    rss_event_results = await asyncio.gather(*rss_event_tasks, return_exceptions=True)

    for result in rss_event_results:
        if isinstance(result, dict) and result: # Ensure result is a non-empty dict
            rss_feeds_collection_list.append(result)

    if not rss_feeds_collection_list:
        logger.warning("No valid RSS events were generated for calibration.")
        return []
    
    # Process correlations concurrently
    tasks = []
    for ev in limited_user_events:
        # Pass the list of RSS feed collections directly
        tasks.append(process_single_correlation(ev, rss_feeds_collection_list)) 
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Filter successful results
    calibration_data = []
    for result in results:
        if isinstance(result, dict):
            calibration_data.append(result)
    
    return calibration_data

async def generate_events_for_single_rss_item(item: dict) -> dict:
    """Helper to generate events for a single RSS item text."""
    try:
        events_json_raw = generate_events_for_text(item["text"]).strip()
        events_json = extract_json_block(events_json_raw)
        # The events_json should be a list of events for this single RSS item
        parsed_events = parse_json_any(events_json)
        if not isinstance(parsed_events, list): # Ensure it's a list
             parsed_events = [parsed_events] if parsed_events else []
        return {
            "rss_source_id": item["entry_id"],
            "rss_events": parsed_events
        }
    except Exception as e:
        logger.error(f"Failed to generate events for RSS item {item.get('entry_id')}: {e}")
        return {} # Return empty dict on failure for this item

async def process_single_correlation(event: dict, rss_feeds_collection_list: list) -> dict:
    """Process correlation for a single event against a list of RSS feed collections."""
    try:
        # generate_event_correlation_single expects a list for rss_feeds_collection
        return generate_event_correlation_single(event, rss_feeds_collection_list)
    except Exception as e:
        logger.error(f"Correlation failed for event {event.get('event_numeric_id')}: {e}")
        return {
                        "user_event_identifier": {
                "event_numeric_id": str(event.get("event_numeric_id", "")),
                "event_summary": event.get("event", ""),
                "event_quote": event.get("quote", "")
                        },
                        "correlation_results": {
                            "is_event_in_rss": False,
                            "exact_matches_in_rss": [],
                            "has_close_by_events_in_rss": False,
                            "close_by_events_details": []
                        }
        }

async def main(input_text: Optional[str] = None, author_name: Optional[str] = None, return_calibration_only: bool = False, model: Optional[SentenceTransformer] = None) -> dict:
    """Main function optimized for speed with two-phase processing."""
    load_dotenv()
    
    # 🔍 DEBUG: Start of main function
    logger.info("🔍 DEBUG: Starting main function")
    logger.info(f"🔍 DEBUG: Input parameters - input_text length: {len(input_text) if input_text else 0}, author_name: {author_name}, return_calibration_only: {return_calibration_only}")
    
    # Use provided model or create new one if not provided
    if model is None:
        logger.info("🔍 DEBUG: Creating new SentenceTransformer model")
        model = SentenceTransformer(MODEL_NAME)
    else:
        logger.info("🔍 DEBUG: Using provided SentenceTransformer model")

    # Get input and handle JSON format with author extraction
    if input_text is None:
        print("Enter your search text. When finished, type DONE:")
        lines = []
        while True:
            line = input()
            if line.strip().upper() == "DONE":
                break
            lines.append(line)
        query = "\n".join(lines).strip()
    else:
        query = input_text
        logger.info(f"🔍 DEBUG: Processing input text: {query[:100]}...")  # Log first 100 chars of input
        
        # Try to parse as JSON to extract author
        try:
            json_input = json.loads(query)
            logger.info(f"🔍 DEBUG: Successfully parsed JSON input: {json_input.keys()}")
            
            if isinstance(json_input, dict):
                if "author" in json_input:
                    author_name = json_input["author"]
                    logger.info(f"🔍 DEBUG: ✅ Extracted author from JSON: {author_name}")
                else:
                    logger.warning("🔍 DEBUG: ❌ No 'author' field found in JSON input")
                
                # Extract the actual text content
                if "text" in json_input:
                    query = json_input["text"]
                    logger.info(f"🔍 DEBUG: ✅ Extracted text from JSON: {query[:100]}...")
                elif "content" in json_input:
                    query = json_input["content"]
                    logger.info("🔍 DEBUG: ✅ Using 'content' field as text")
                elif "query" in json_input:
                    query = json_input["query"]
                    logger.info("🔍 DEBUG: ✅ Using 'query' field as text")
                else:
                    # If no specific text field, use the whole JSON as string
                    query = str(json_input)
                    logger.warning("🔍 DEBUG: ⚠️ No specific text field found, using entire JSON as text")
        except json.JSONDecodeError:
            logger.warning("🔍 DEBUG: ⚠️ Input is not valid JSON, using as plain text")
            # Not JSON, use as-is
            pass

    if author_name:
        logger.info(f"🔍 DEBUG: ✅ Adding author context to query: {author_name}")
        query = f"Author: {author_name}\n\n{query}"
    else:
        logger.warning("🔍 DEBUG: ⚠️ No author name available")

    logger.info(f"🔍 DEBUG: Final query text: {query[:200]}...")

    # Process all data concurrently with optimizations
    logger.info("🔍 DEBUG: Starting concurrent processing")
    
    # Start all major processing tasks concurrently
    logger.info("🔍 DEBUG: 📡 Starting RSS data processing...")
    rss_task = process_rss_data(query, model)
    
    logger.info("🔍 DEBUG: 📝 Starting user events processing...")
    events_task = process_user_events(query)
    
    # Wait for initial results
    logger.info("🔍 DEBUG: ⏳ Waiting for RSS and events processing to complete...")
    (rss_payload, matches_dict), (user_events, user_events_json_raw) = await asyncio.gather(
        rss_task, events_task
    )
    
    # 🔍 DEBUG: Log results from first phase
    logger.info(f"🔍 DEBUG: 📡 RSS processing complete - Found {len(rss_payload)} RSS articles")
    logger.info(f"🔍 DEBUG: 📡 RSS matches found: {len([m for m in matches_dict.values() if m.get('Match')])}")
    logger.info(f"🔍 DEBUG: 📝 User events extracted: {len(user_events)}")
    
    # Log sample RSS data
    if rss_payload:
        logger.info(f"🔍 DEBUG: 📡 Sample RSS article: {rss_payload[0].get('entry_id', 'No ID')[:100]}...")
    
    # Log sample user events
    if user_events:
        logger.info(f"🔍 DEBUG: 📝 Sample user event: {user_events[0].get('event', 'No event')}")
    
    # Start Wikipedia and calibration processing
    logger.info("🔍 DEBUG: 📚 Starting Wikipedia data processing...")
    wiki_task = process_wikipedia_data(user_events_json_raw, query, author_name)
    
    logger.info("🔍 DEBUG: 🔗 Starting calibration data generation...")
    calibration_task = generate_calibration_data(user_events, rss_payload)
    
    # Wait for calibration and wiki results
    logger.info("🔍 DEBUG: ⏳ Waiting for Wikipedia and calibration processing to complete...")
    (char_info, author_info), calibration_data = await asyncio.gather(
        wiki_task, calibration_task
    )
    
    # 🔍 DEBUG: Log results from second phase
    logger.info(f"🔍 DEBUG: 📚 Wikipedia processing complete - Found {len(char_info)} characters")
    logger.info(f"🔍 DEBUG: 📚 Author info: {author_info.get('name', 'Unknown')}")
    logger.info(f"🔍 DEBUG: 🔗 Calibration data generated: {len(calibration_data)} correlations")
    
    # Log sample character info
    if char_info:
        sample_char = list(char_info.keys())[0]
        logger.info(f"🔍 DEBUG: 📚 Sample character: {sample_char}")
    
    # Log sample calibration data
    if calibration_data:
        sample_cal = calibration_data[0]
        logger.info(f"🔍 DEBUG: 🔗 Sample calibration: Event {sample_cal.get('user_event_identifier', {}).get('event_numeric_id', 'Unknown')}")
    
    # Create calibration report (Phase 1)
    calibration_report = {
        "phase": "calibration",
        "query": query,
        "author": author_name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "author_info": author_info,
        "summary": {
            "total_events_analyzed": len(user_events),
            "rss_matches": len([m for m in matches_dict.values() if m.get("Match")]),
            "wikipedia_characters": len(char_info),
            "calibration_events": len(calibration_data)
        },
        "user_events": user_events,
        "rss_payload": rss_payload,
        "matches_dict": matches_dict,
        "character_info": char_info,
        "calibration_data": calibration_data
    }
    
    # Save calibration data
    with open("calibration_output.json", "w", encoding="utf-8") as f:
        json.dump(calibration_report, f, ensure_ascii=False, indent=2)
    
    logger.info("🔍 DEBUG: ✔ Phase 1 complete: Calibration data saved to calibration_output.json")
    
    # If only calibration requested, return early
    if return_calibration_only:
        logger.info("🔍 DEBUG: 🔄 Returning calibration-only results")
        return calibration_report
    
    # Phase 2: Generate fact-check report
    logger.info("🔍 DEBUG: 🎯 Starting Phase 2: Fact-checking analysis")
    
    fact_check_results: list = []
    
    # Map user_event_id → characters for wiki lookup
    event_char_map = {str(ev.get("event_numeric_id")): ev.get("characters", [])
                      for ev in user_events}
    
    logger.info(f"🔍 DEBUG: 🎯 Event-character mapping created: {len(event_char_map)} events")
    
    # Process fact checks concurrently with batching for better performance
    fact_check_tasks = []
    
    for obj in sorted(calibration_data, key=lambda o: int(o['user_event_identifier']['event_numeric_id'])):
        user_id_str = obj["user_event_identifier"]["event_numeric_id"]
        user_id = int(user_id_str)  # Convert to int
        user_quote = obj.get("user_event_identifier", {}).get("event_quote", "") or ""
        
        logger.info(f"🔍 DEBUG: 🎯 Preparing fact-check for event {user_id}: {user_quote[:50]}...")
        
        # RSS quotes (exact matches only, up to 2)
        rss_quotes = []
        for m in obj.get("correlation_results", {}).get("exact_matches_in_rss", [])[:2]:
            rss_quotes.append({
                "rss_source_id": m.get("rss_source_id", ""),
                "quote": m.get("matched_rss_event", {}).get("quote", "")
            })
        
        # Wikipedia quotes (up to 2)
        wiki_quotes = []
        for char in event_char_map.get(str(user_id), [])[:2]:
            if char in char_info:
                wiki_quotes.append({
                    "character": char,
                    "quote": char_info[char].get("quote", ""),
                    "url": char_info[char].get("url", "")
                })
        
        logger.info(f"🔍 DEBUG: 🎯 Event {user_id} - RSS quotes: {len(rss_quotes)}, Wiki quotes: {len(wiki_quotes)}")
        
        # Create fact check task
        fact_check_tasks.append(
            process_single_fact_check(
                user_id, user_quote, rss_quotes, wiki_quotes, obj,
                query, author_name or "", author_info, user_events, rss_payload, matches_dict, char_info, calibration_data
            )
        )
    
    logger.info(f"🔍 DEBUG: 🎯 Created {len(fact_check_tasks)} fact-check tasks")
    
    # Execute fact checks in batches for better performance
    if fact_check_tasks:
        batch_size = 5  # Process 5 fact checks at a time to avoid overwhelming the API
        fact_check_results = []
        
        logger.info(f"🔍 DEBUG: 🎯 Processing fact-checks in batches of {batch_size}")
        
        for i in range(0, len(fact_check_tasks), batch_size):
            batch = fact_check_tasks[i:i + batch_size]
            logger.info(f"🔍 DEBUG: 🎯 Processing batch {i//batch_size + 1}/{(len(fact_check_tasks) + batch_size - 1)//batch_size}")
            
            batch_results = await asyncio.gather(*batch, return_exceptions=True)
            
            # Filter successful results
            for j, result in enumerate(batch_results):
                if isinstance(result, dict):
                    fact_check_results.append(result)
                    logger.info(f"🔍 DEBUG: 🎯 Batch {i//batch_size + 1}, Task {j+1}: Success")
                else:
                    logger.error(f"🔍 DEBUG: 🎯 Batch {i//batch_size + 1}, Task {j+1}: Failed - {result}")
            
            # Small delay between batches to be respectful to the API
            if i + batch_size < len(fact_check_tasks):
                await asyncio.sleep(0.5)
    
    logger.info(f"🔍 DEBUG: 🎯 Fact-checking complete - {len(fact_check_results)} successful results")
    
    # Create final report (Phase 2)
    final_report = {
        "phase": "final",
        "query": query,
        "author": author_name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "author_info": author_info,
        "summary": {
            "total_events_analyzed": len(user_events),
            "fact_checks_completed": len(fact_check_results),
            "rss_matches": len([m for m in matches_dict.values() if m.get("Match")]),
            "wikipedia_characters": len(char_info)
        },
        "fact_check_results": fact_check_results
    }
    
    # Save final output
    with open("fact_check_output.json", "w", encoding="utf-8") as f:
        json.dump(final_report, f, ensure_ascii=False, indent=2)
    
    logger.info("🔍 DEBUG: ✔ Phase 2 complete: Final fact-check analysis saved to fact_check_output.json")
    logger.info(f"🔍 DEBUG: 🏁 FINAL SUMMARY:")
    logger.info(f"🔍 DEBUG: 🏁 - Events analyzed: {len(user_events)}")
    logger.info(f"🔍 DEBUG: 🏁 - RSS matches: {len([m for m in matches_dict.values() if m.get('Match')])}")
    logger.info(f"🔍 DEBUG: 🏁 - Wikipedia characters: {len(char_info)}")
    logger.info(f"🔍 DEBUG: 🏁 - Fact-checks completed: {len(fact_check_results)}")
    
    return final_report

# Add a new function for API endpoints to use
async def run_calibration_phase(input_text: str, model: Optional[SentenceTransformer] = None) -> dict:
    """Run only the calibration phase and return results."""
    return await main(input_text=input_text, return_calibration_only=True, model=model)

async def run_full_analysis(input_text: str, model: Optional[SentenceTransformer] = None) -> dict:
    """Run the complete analysis (both phases)."""
    return await main(input_text=input_text, return_calibration_only=False, model=model)

async def process_single_fact_check(
    user_id: int, 
    user_quote: str, 
    rss_quotes: list, 
    wiki_quotes: list, 
    correlation_obj: dict,
    # Additional comprehensive data
    query: str,
    author: str,
    author_info: dict,
    user_events: list,
    rss_payload: list,
    matches_dict: dict,
    character_info: dict,
    calibration_data: list
) -> dict:
    """Process a single fact-check with comprehensive context and search capabilities."""
    logger.info(f"🔍 DEBUG: 🎯 Processing fact-check for event {user_id}")
    logger.info(f"🔍 DEBUG: 🎯 User quote: {user_quote[:100]}...")
    logger.info(f"🔍 DEBUG: 🎯 RSS quotes available: {len(rss_quotes)}")
    logger.info(f"🔍 DEBUG: 🎯 Wiki quotes available: {len(wiki_quotes)}")
    
    try:
        # Prepare data for LLM
        rss_quotes_json = json.dumps(rss_quotes, ensure_ascii=False)
        wiki_quotes_json = json.dumps(wiki_quotes, ensure_ascii=False)
        correlation_json = json.dumps(correlation_obj, ensure_ascii=False)
        
        # Additional comprehensive context
        author_info_json = json.dumps(author_info, ensure_ascii=False)
        user_events_json = json.dumps(user_events, ensure_ascii=False)
        rss_payload_json = json.dumps(rss_payload, ensure_ascii=False)
        matches_dict_json = json.dumps(matches_dict, ensure_ascii=False)
        character_info_json = json.dumps(character_info, ensure_ascii=False)
        calibration_data_json = json.dumps(calibration_data, ensure_ascii=False)
        
        logger.info(f"🔍 DEBUG: 🎯 Prepared JSON data for LLM - correlation: {len(correlation_json)} chars")
        
        # Call the enhanced LLM fact checker with search capabilities
        logger.info(f"🔍 DEBUG: 🎯 Calling enhanced LLM fact checker for event {user_id}...")
        
        # Import the enhanced fact checker
        from llm.llm_fact_checker import generate_fact_check_with_search
        
        fact_check_result = await generate_fact_check_with_search(
            user_id, user_quote, rss_quotes_json, wiki_quotes_json, correlation_json,
            query, author, author_info_json, user_events_json, rss_payload_json, 
            matches_dict_json, character_info_json, calibration_data_json
        )
        
        logger.info(f"🔍 DEBUG: 🎯 Enhanced LLM fact check complete for event {user_id}")
        logger.info(f"🔍 DEBUG: 🎯 Result length: {len(fact_check_result)} characters")
        
        # Return the plain text result directly
        return {
            "user_event_id": user_id,
            "user_event_quote": user_quote,
            "analysis": fact_check_result  # Plain text response
        }
    
    except Exception as e:
        logger.error(f"🔍 DEBUG: 🎯 Fact check failed for event {user_id}: {e}")
        return {
            "user_event_id": user_id,
            "user_event_quote": user_quote,
            "analysis": f"שגיאה בבדיקת עובדות: {str(e)}"
        }

if __name__ == "__main__":
    result = asyncio.run(main())
    print(f"✅ Analysis complete! Results saved to output files.") 