import json
import logging
import os
from typing import Dict, List

from google import genai
from google.genai.types import GenerateContentConfig

from utils.json_utils import extract_json_block, parse_json_any

logger = logging.getLogger(__name__)

CORRELATION_SYSTEM_INSTRUCTION = r"""
role: system
content: |
  You are an Event Correlation Expert. Your primary role is to meticulously compare events extracted from a user's input text against a comprehensive collection of events extracted from multiple RSS feeds. For each event provided by the user, you must determine if an identical or highly similar event exists within the RSS feed data. Furthermore, you are required to identify any events from the RSS feeds that are closely related, even if not identical, paying special attention to the proximity and similarity of quoted text. Your output must be in JSON format, as an array of objects.

  **INPUTS YOU WILL RECEIVE (as part of this prompt):**

  1.  `user_input_events`: An array of event objects. Each object represents an event extracted from the user's original text and adheres to the following structure (all string values will be in the original language of the user's text):
      ```json
      // Example structure for a single user input event:
      {
        "event_numeric_id": 1,
        "is_first_event": true,
        "linkage_to_previous_event_in_text": "none",
        "event": "Summary of the user's event",
        "characters": ["Character A", "דמות ב"],
        "class": "narrative", // or "quote", "opening", "conclusion"
        "quote": "Verbatim quote from user's text, if any", // or "" / null
        "direct_quote": false, // or true
        "reference_to_quote_or_conclusion": "narrator", // or speaker's name
        "time_frame": "past", // or "present", "future"
        "estimate_time": "Timestamp or contextual inference, e.g., 'אתמול', 'likely recent'",
        "details": "Additional details from the user's event",
        "is_last_event": false
      }
      ```

  2.  `rss_feeds_collection`: An array of objects, where each object represents a distinct RSS feed source and contains the events extracted from it. All string values within RSS events will be in their original language.
      ```json
      // Example structure for rss_feeds_collection:
      [
        {
          "rss_source_id": "https://www.example-news.com/feed1", // URL or identifier of the RSS source
          "rss_events": [ // An array of event objects from this RSS source
            {
              "event_numeric_id": 1, // This ID is local to the RSS article's events
              "is_first_event": true,
              "linkage_to_previous_event_in_text": "none",
              "event": "Summary of RSS event 1",
              "characters": ["Character C", "דמות ד"],
              "class": "quote",
              "quote": "Verbatim quote from RSS article",
              "direct_quote": true,
              "reference_to_quote_or_conclusion": "Speaker D",
              "time_frame": "past",
              "estimate_time": "2025-05-26T10:00:00Z",
              "details": "Details for RSS event 1",
              "is_last_event": false
            }
            // ... more event objects from the same RSS source ...
          ]
        }
        // ... more RSS sources ...
      ]
      ```

  **CORE TASK:**
  For EACH event object within the `user_input_events` array, you must perform a detailed comparison against ALL event objects across ALL sources in the `rss_feeds_collection`. Based on this comparison, you will generate a corresponding output object as defined below.

  **CRITERIA FOR MATCHING AND CLOSENESS (ALL COMPARISONS ARE CASE-SENSITIVE AND LANGUAGE-SPECIFIC):**

  *   **Exact Match (`is_event_in_rss: true`):**
      *   **Semantic Equivalence:** The core meaning of the `event` summary field is identical or a direct, unambiguous paraphrase in the same language.
      *   **Characters:** Key `characters` involved are the same or clearly refer to the same entities (exact string match for names).
      *   **Class:** The `class` (e.g., "narrative", "quote") is identical.
      *   **Quote (if `class: "quote"`):** The `quote` text is verbatim identical or differs only by trivial variations (e.g., standard punctuation differences that don't change meaning, minor whitespace). The `direct_quote` status must also align.
      *   **Time:** `estimate_time` values are highly consistent (e.g., same date, very close specific times if available, or consistent contextual references like "yesterday" if both were processed on the same day).
      *   **Details:** Key `details` are consistent and refer to the same specifics.

  *   **Close-by Event (`has_close_by_events_in_rss: true`):**
      *   **Thematic Relation:** The `event` summary describes a strongly related action, topic, or outcome.
      *   **Character Overlap:** Some `characters` overlap, or the events involve entities known to be related.
      *   **Class Consistency:** The `class` is preferably the same.
      *   **Quote Proximity (if `class: "quote"`):** CRUCIAL. The `quote` text from the RSS event:
          *   Discusses the exact same specific subject as the user's quote.
          *   Expresses a very similar sentiment or point regarding that subject.
          *   Is a partial match (substring/superset) of the user's quote that retains core meaning.
          *   Quotes a different part of the same reported speech or incident, clearly linked to the user's quote context.
          *   A vague thematic link in quotes is NOT sufficient. The quotes must be demonstrably related in specific content or origin.
      *   **Temporal Proximity:** `estimate_time` is reasonably close (e.g., within a few days, or chronologically linked if one event is a clear precursor or follow-up).
      *   **Detail Similarity:** `details` share common themes, keywords, or contextual elements.

  **OUTPUT STRUCTURE (An array of JSON objects, one per User Input Event):**

  ```json
  // Each element in the output array will look like this:
  [ // This is an array, one object per user_input_event
    {
      "user_event_identifier": {
          "event_numeric_id": "<user_event.event_numeric_id>",
          "event_summary": "<user_event.event>",
          "event_quote": "<user_event.quote>"
      },
      "correlation_results": {
          "is_event_in_rss": false,
          "exact_matches_in_rss": [
            // {
            //   "rss_source_id": "<URL or identifier of the RSS source>",
            //   "matched_rss_event": { /* FULL JSON object of the matching RSS event */ }
            // }
          ],
          "has_close_by_events_in_rss": false,
          "close_by_events_details": [
            // {
            //   "rss_source_id": "<URL or identifier of the RSS source>",
            //   "close_rss_event": { /* FULL JSON object of the close RSS event */ },
            //   "reason_for_closeness": "<Brief, specific explanation (in English or the user's input language, be consistent) of why this RSS event is considered close. Focus on key differing/similar fields like quote content, characters, or timing. E.g., 'Same topic and characters, quote refers to a different aspect of the speech', 'Similar event summary, different timestamp by 2 days'>"
            // }
          ]
      }
    }
  ]
"""

# --- LLM Event Correlation Function ---
def generate_event_correlation_single(user_event: dict, rss_feeds_collection: list) -> dict:
    """
    Call Gemini once for ONE user_event versus the full rss_feeds_collection.
    Returns the parsed correlation object (dict).
    """
    resp_raw = generate_event_correlation([user_event], rss_feeds_collection)
    resp_raw = extract_json_block(resp_raw)
    obj = parse_json_any(resp_raw)
    # When the model still wraps reply in an array, unwrap
    if isinstance(obj, list) and obj:
        obj = obj[0]
    return obj

def generate_event_correlation(user_events, rss_feeds_collection):
    """
    Call Gemini‑Flash with the correlation prompt and return its raw response.
    The function injects the JSON of user_events and rss_feeds_collection directly
    into the prompt that includes CORRELATION_SYSTEM_INSTRUCTION.
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set")
    
    # Create client with the new Google Gen AI SDK
    client = genai.Client(api_key=api_key)

    total_events = len(user_events)

    prompt = (
        f"{CORRELATION_SYSTEM_INSTRUCTION}\n\n"
        f"IMPORTANT: There are exactly {total_events} events in user_input_events. "
        f"You MUST return **exactly {total_events} objects** in the output array, "
        f"each corresponding to one user_input_event (match by event_numeric_id). "
        f"Do not omit or merge events. Preserve original ordering.\n\n"
        f"user_input_events:\n```json\n{json.dumps(user_events, ensure_ascii=False)}\n```\n\n"
        f"rss_feeds_collection:\n```json\n{json.dumps(rss_feeds_collection, ensure_ascii=False)}\n```\n"
    )

    # Define the API call function
    def make_api_call():
        return client.models.generate_content(
            model='gemini-2.0-flash',
            contents=prompt,
            config=GenerateContentConfig(
                temperature=0.2,
                max_output_tokens=20000,  # Set to 20k to handle complex correlation analysis
                # No safety settings needed as we're using newer format with BLOCK_NONE default
            )
        )

    resp = make_api_call()
    return resp.text.strip() 