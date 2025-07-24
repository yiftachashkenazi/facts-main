#!/usr/bin/env python3
# File: wiki/wiki_processor.py
"""
Wikipedia processing module for fact-checking application.
- Fetches Wikipedia data for characters mentioned in events.
- Provides author information lookup functionality.
- Returns structured data for the main pipeline.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())
import asyncio
import re

import aiohttp
import wikipedia
from bs4 import BeautifulSoup
from google import genai

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
# Helpers
# ——————————————————————————————————————————— #
def fetch_wiki_data(entity_name: str) -> Dict:
    """Fetch Wikipedia summary and full text for an entity using the wikipedia library."""
    try:
        page = wikipedia.page(entity_name)
        summary = page.summary or ""
        full_text = page.content or ""
        url = page.url or ""
    except Exception as e:
        logger.warning(f"Wikipedia lookup failed for {entity_name}: {e}")
        return {"entity_name": entity_name, "summary": "", "full_text": "", "url": ""}
    return {
        "entity_name": entity_name,
        "summary": summary[:1000],
        "full_text": full_text[:5000],
        "url": url
    }

# --- LLM Wikipedia Processing Function ---
async def process_characters_wiki(events_json: str, query: str) -> Dict[str, Any]:
    """
    Process Wikipedia data for characters mentioned in events.
    
    Args:
        events_json (str): JSON string containing events with character information
        query (str): The original query text
        
    Returns:
        Dict[str, Any]: Dictionary mapping character names to their Wikipedia information
    """
    try:
        # Parse events JSON
        events = parse_json_any(events_json)
        if not isinstance(events, list):
            events = [events]
            
        # Extract unique character names
        characters = set()
        for ev in events:
            characters.update(ev.get("characters", []))
            
        if not characters:
            return {}
            
        # Process each character asynchronously
        async def process_character(char: str) -> tuple[str, Dict[str, Any]]:
            try:
                # Search Wikipedia for the character
                search_results = wikipedia.search(char, results=3)
                if not search_results:
                    return char, {
                        "quote": f"No Wikipedia entry found for {char}",
                        "url": "",
                        "summary": "",
                        "content": ""
                    }
                
                # Get the first result's page
                page = wikipedia.page(search_results[0])
                
                # Extract relevant information
                summary = page.summary[:500]  # First 500 characters of summary
                content = page.content[:2000]  # First 2000 characters of content
                
                # Find a relevant quote from the content
                quote = summary
                if len(content) > len(summary):
                    # Try to find a sentence that mentions the character
                    sentences = content.split(". ")
                    for sentence in sentences:
                        if char.lower() in sentence.lower():
                            quote = sentence
                            break
                
                return char, {
                    "quote": quote,
                    "url": page.url,
                    "summary": summary,
                    "content": content
                }
                
            except wikipedia.exceptions.DisambiguationError:
                return char, {
                    "quote": f"Multiple Wikipedia entries found for {char}, unable to determine correct one",
                    "url": "",
                    "summary": "",
                    "content": ""
                }
            except wikipedia.exceptions.PageError:
                return char, {
                    "quote": f"No Wikipedia entry found for {char}",
                    "url": "",
                    "summary": "",
                    "content": ""
                }
            except Exception as exc:
                logger.error("Failed to process character %s: %s", char, exc)
                return char, {
                    "quote": f"Error retrieving information for {char}",
                    "url": "",
                    "summary": "",
                    "content": ""
                }
        
        # Process all characters concurrently
        tasks = [process_character(char) for char in characters]
        results = await asyncio.gather(*tasks)
        
        # Convert results to dictionary
        return dict(results)
        
    except Exception as exc:
        logger.error("Failed to process Wikipedia data: %s", exc)
        return {}

async def get_author_info_by_name(author_name: str) -> Dict[str, Any]:
    """
    Get author information from Wikipedia using the provided author name directly.
    
    Args:
        author_name (str): The author's name to look up
        
    Returns:
        Dict[str, Any]: Dictionary containing author information
    """
    logger.info(f"Starting author info extraction for author: {author_name}")
    
    try:
        if not author_name or author_name.strip() == "":
            logger.warning("❌ Empty author name provided")
            return {
                "name": "Unknown",
                "background": "No author name provided",
                "reliability_score": 0.0,
                "bias_analysis": "Unable to analyze bias without author information"
            }
        
        # Clean the author name
        clean_author_name = author_name.strip()
        logger.info(f"🔍 Searching Wikipedia for author: {clean_author_name}")
        
        # Detect language and set Wikipedia language
        has_hebrew = bool(re.search(r'[\u0590-\u05FF]', clean_author_name))
        lang = "he" if has_hebrew else "en"
        
        logger.info(f"🔍 Setting Wikipedia language to: {lang}")
        wikipedia.set_lang(lang)
        
        # Search Wikipedia for author information
        try:
            # Search Wikipedia for the author
            search_results = wikipedia.search(clean_author_name, results=5)
            if not search_results:
                logger.warning(f"❌ No Wikipedia results found for author: {clean_author_name}")
                
                # Try with the other language if no results found
                other_lang = "en" if lang == "he" else "he"
                logger.info(f"🔍 Trying with {other_lang} Wikipedia...")
                wikipedia.set_lang(other_lang)
                search_results = wikipedia.search(clean_author_name, results=5)
                
                if not search_results:
                    return {
                        "name": clean_author_name,
                        "background": "No Wikipedia entry found",
                        "reliability_score": 0.5,
                        "bias_analysis": "Unable to determine bias without background information"
                    }
            
            logger.info(f"✅ Found Wikipedia results for {clean_author_name}: {search_results}")
            
            # Try each search result until we find a good match
            page = None
            for result in search_results:
                try:
                    page = wikipedia.page(result)
                    logger.info(f"✅ Retrieved Wikipedia page: {page.title}")
                    break
                except wikipedia.exceptions.PageError:
                    logger.warning(f"⚠️ Page error for result: {result}, trying next...")
                    continue
                except wikipedia.exceptions.DisambiguationError as e:
                    logger.warning(f"⚠️ Disambiguation for result: {result}, trying first option...")
                    try:
                        page = wikipedia.page(e.options[0])
                        logger.info(f"✅ Retrieved disambiguation page: {page.title}")
                        break
                    except:
                        continue
            
            if not page:
                logger.warning(f"❌ Could not retrieve any Wikipedia page for {clean_author_name}")
                return {
                    "name": clean_author_name,
                    "background": "Wikipedia page could not be retrieved",
                    "reliability_score": 0.3,
                    "bias_analysis": "Unable to determine bias without accessible page"
                }
            
            # Extract relevant information
            background = page.summary[:500]  # First 500 characters of summary
            logger.info(f"✅ Extracted background info: {background[:100]}...")
            
            # Analyze reliability based on Wikipedia content (support both languages)
            reliability_score = 0.5  # Base score
            background_lower = background.lower()
            
            reliability_keywords = [
                "expert", "award-winning", "renowned", "respected", "scholar", "professor", # English
                "מומחה", "זוכה פרסים", "נודע", "מכובד", "חוקר", "פרופסור" # Hebrew
            ]
            bias_keywords = [
                "activist", "controversial", "criticized", "partisan", "affiliated", "opinionated", # English
                "פעיל", "שנוי במחלוקת", "ספג ביקורת", "מפלגתי", "מזוהה עם", "בעל דעה" # Hebrew
            ]
            
            if any(keyword in background_lower for keyword in reliability_keywords):
                reliability_score = min(1.0, reliability_score + 0.2)
                logger.info("✅ Increased reliability due to keywords")
            if any(keyword in background_lower for keyword in bias_keywords):
                reliability_score = max(0.0, reliability_score - 0.2)
                logger.info("⚠️ Decreased reliability due to bias keywords")
            
            logger.info(f"📊 Calculated reliability score: {reliability_score}")
            
            # Detect language for bias analysis
            has_hebrew = bool(re.search(r'[\u0590-\u05FF]', background))
            lang = "he" if has_hebrew else "en"
            
            # Generate bias analysis using Gemini
            api_key = os.environ.get("GEMINI_API_KEY")
            if not api_key:
                logger.error("GEMINI_API_KEY environment variable not set")
                # Fallback if API key is not available
                return {
                    "name": clean_author_name,
                    "background": background,
                    "reliability_score": reliability_score,
                    "bias_analysis": "API key not found, unable to perform bias analysis."
                }
            
            model = genai.GenerativeModel("gemini-2.0-flash", api_key=api_key) # Pass API key here
            prompt = f"""Analyze the potential bias of an author based on this Wikipedia summary. 
                        Focus on political leaning, affiliations, and controversial statements. 
                        Be concise and objective. Author: {clean_author_name}. Summary: {background}
                        Provide the analysis in {"Hebrew" if lang == "he" else "English"}."""
            
            response = model.generate_content(prompt)
            bias_analysis = response.text.strip()
            logger.info(f"🤖 Generated bias analysis: {bias_analysis[:100]}...")
            
            return {
                "name": clean_author_name,
                "background": background,
                "reliability_score": reliability_score,
                "bias_analysis": bias_analysis,
                "wikipedia_title": page.title,
                "wikipedia_url": page.url,
                "language": lang
            }
            
        except wikipedia.exceptions.DisambiguationError as e:
            logger.warning(f"❌ Multiple Wikipedia entries found for {clean_author_name}")
            # Try to find the most relevant one
            options = e.options[:5]  # Take first 5 options
            logger.info(f"Available options: {options}")
            
            # Try the first few options to see if any work
            for option in options[:3]:
                try:
                    page = wikipedia.page(option)
                    background = page.summary[:500]
                    logger.info(f"✅ Using disambiguation option: {page.title}")
                    
                    return {
                        "name": clean_author_name,
                        "background": background,
                        "reliability_score": 0.6,
                        "bias_analysis": f"Found via disambiguation: {page.title}",
                        "wikipedia_title": page.title,
                        "wikipedia_url": page.url,
                        "disambiguation_options": options
                    }
                except:
                    continue
            
            return {
                "name": clean_author_name,
                "background": f"Multiple Wikipedia entries found: {', '.join(options[:3])}",
                "reliability_score": 0.3,
                "bias_analysis": "Unable to determine bias due to ambiguous author identity",
                "disambiguation_options": options
            }
        except wikipedia.exceptions.PageError:
            logger.warning(f"❌ No Wikipedia page found for {clean_author_name}")
            return {
                "name": clean_author_name,
                "background": "No Wikipedia entry found",
                "reliability_score": 0.5,
                "bias_analysis": "Unable to determine bias without background information"
            }
            
    except Exception as exc:
        logger.error(f"❌ Failed to get author information for {author_name}: {exc}")
        return {
            "name": author_name,
            "background": "Error retrieving author information",
            "reliability_score": 0.0,
            "bias_analysis": "Unable to analyze bias due to error"
        }

async def get_author_info(text: str) -> Dict[str, Any]:
    """
    Extract author name using Gemini and then get information from Wikipedia.

    Args:
        text (str): The input text containing a potential author mention.

    Returns:
        Dict[str, Any]: Dictionary containing author information, or default if not found.
    """
    logger.info(f"Starting author extraction for text: {text[:50]}...")
    
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        logger.error("GEMINI_API_KEY environment variable not set")
        return {
            "name": "Unknown",
            "background": "API key for Gemini not found, cannot extract author.",
            "reliability_score": 0.0,
            "bias_analysis": "Author extraction failed due to missing API key."
        }

    model = genai.GenerativeModel("gemini-2.0-flash", api_key=api_key) # Pass API key here

    prompt = (
        "Extract the full name of the author or primary source (person or organization) " 
        f"from the following text: {text}. If no author is mentioned, return 'Unknown'."
    )

    response = model.generate_content(prompt)
    author_name = response.text.strip()

    if author_name.lower() == "unknown":
        logger.warning("❌ No author name found in text")
        return {
            "name": "Unknown",
            "background": "No author information found",
            "reliability_score": 0.0,
            "bias_analysis": "Unable to analyze bias without author information"
        }

    logger.info(f"✅ Extracted author name: {author_name}")

    # Search Wikipedia for author information
    try:
        # Search Wikipedia for the author
        logger.info(f"🔍 Searching Wikipedia for author: {author_name}")
        search_results = wikipedia.search(author_name, results=3)
        if not search_results:
            logger.warning(f"❌ No Wikipedia results found for author: {author_name}")
            return {
                "name": author_name,
                "background": "No Wikipedia entry found",
                "reliability_score": 0.5,
                "bias_analysis": "Unable to determine bias without background information"
            }
        
        logger.info(f"✅ Found Wikipedia results for {author_name}: {search_results}")
        
        # Get the first result's page
        page = wikipedia.page(search_results[0])
        logger.info(f"✅ Retrieved Wikipedia page: {page.title}")
        
        # Extract relevant information
        background = page.summary[:500]  # First 500 characters of summary
        logger.info(f"✅ Extracted background info: {background[:100]}...")
        
        # Analyze reliability based on Wikipedia content
        reliability_score = 0.5  # Base score
        background_lower = background.lower()
        
        reliability_keywords = [
            "expert", "award-winning", "renowned", "respected", "scholar", "professor", # English
            "מומחה", "זוכה פרסים", "נודע", "מכובד", "חוקר", "פרופסור" # Hebrew
        ]
        bias_keywords = [
            "activist", "controversial", "criticized", "partisan", "affiliated", "opinionated", # English
            "פעיל", "שנוי במחלוקת", "ספג ביקורת", "מפלגתי", "מזוהה עם", "בעל דעה" # Hebrew
        ]
        
        if any(keyword in background_lower for keyword in reliability_keywords):
            reliability_score = min(1.0, reliability_score + 0.2)
            logger.info("✅ Increased reliability due to keywords")
        if any(keyword in background_lower for keyword in bias_keywords):
            reliability_score = max(0.0, reliability_score - 0.2)
            logger.info("⚠️ Decreased reliability due to bias keywords")
        
        logger.info(f"📊 Calculated reliability score: {reliability_score}")
        
        # Detect language for bias analysis
        has_hebrew = bool(re.search(r'[\u0590-\u05FF]', background))
        lang = "he" if has_hebrew else "en"
        
        # Generate bias analysis using Gemini
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            logger.error("GEMINI_API_KEY environment variable not set")
            # Fallback if API key is not available
            return {
                "name": author_name,
                "background": background,
                "reliability_score": reliability_score,
                "bias_analysis": "API key not found, unable to perform bias analysis."
            }
        
        model = genai.GenerativeModel("gemini-2.0-flash", api_key=api_key) # Pass API key here
        prompt = f"""Analyze the potential bias of an author based on this Wikipedia summary. 
                    Focus on political leaning, affiliations, and controversial statements. 
                    Be concise and objective. Author: {author_name}. Summary: {background}
                    Provide the analysis in {"Hebrew" if lang == "he" else "English"}."""
        
        response = model.generate_content(prompt)
        bias_analysis = response.text.strip()
        logger.info(f"🤖 Generated bias analysis: {bias_analysis[:100]}...")
        
        return {
            "name": author_name,
            "background": background,
            "reliability_score": reliability_score,
            "bias_analysis": bias_analysis
        }
        
    except wikipedia.exceptions.PageError:
        logger.warning(f"❌ Wikipedia page not found for author: {author_name}")
        return {
            "name": author_name,
            "background": "No Wikipedia entry found",
            "reliability_score": 0.5,  # Neutral score for not found
            "bias_analysis": "Unable to determine bias without background information"
        }
    except wikipedia.exceptions.DisambiguationError as e:
        logger.warning(f"❌ Disambiguation error for author: {author_name}: {e}")
        # Attempt to list options or provide a general message
        options = e.options[:3] # Show first 3 options
        options_str = ", ".join(options)
        return {
            "name": author_name,
            "background": f"Multiple Wikipedia entries found (e.g., {options_str}). Please be more specific.",
            "reliability_score": 0.4, # Slightly lower due to ambiguity
            "bias_analysis": "Ambiguous author name; multiple entries found."
        }
    except Exception as exc:
        logger.error(f"❌ Unexpected error processing author {author_name}: {exc}")
        return {
            "name": author_name,
            "background": f"Error retrieving information: {str(exc)}",
            "reliability_score": 0.2, # Low score due to error
            "bias_analysis": "Error during information retrieval prevented bias analysis."
        }

if __name__ == "__main__":
    # Simple test - removed process_wiki_with_grok test since function was deleted
    print("Wiki processor module loaded successfully")
