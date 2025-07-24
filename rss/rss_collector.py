import logging
from typing import Dict, Any
import json

logger = logging.getLogger(__name__)

def build_rss_feeds_collection(rss_events_raw: Dict[str, str]) -> Dict[str, Any]:
    """
    Build a collection of RSS feeds from raw event data.
    
    Args:
        rss_events_raw: Dictionary mapping RSS entry IDs to their raw event JSON strings
        
    Returns:
        Dictionary containing processed RSS feed data organized by entry ID
    """
    collection = {}
    
    for entry_id, raw_json in rss_events_raw.items():
        try:
            # Parse the raw JSON string into a Python object
            events_data = json.loads(raw_json)
            
            # Store the parsed data in the collection
            collection[entry_id] = {
                "raw_json": raw_json,
                "parsed_events": events_data,
                "entry_id": entry_id
            }
        except Exception as e:
            logger.error(f"Failed to process RSS entry {entry_id}: {str(e)}")
            continue
            
    return collection 