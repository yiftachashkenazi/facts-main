import json
import logging
from typing import Any, Dict, List, Optional

from utils.json_utils import parse_json_any

logger = logging.getLogger(__name__)

def build_rss_feeds_collection(rss_events_raw: dict) -> list:
    """
    Convert rss_events_raw {para_id: events_json_str} into the structure required
    for the correlation prompt: a list of {rss_source_id, rss_events}.
    The rss_source_id is the part of para_id before the first '#'.
    
    Args:
        rss_events_raw (dict): Dictionary with para_id as keys and JSON strings as values
        
    Returns:
        list: List of dictionaries with rss_source_id and rss_events
    """
    feeds = {}
    total_entries = len(rss_events_raw)
    processed_entries = 0
    skipped_entries = 0
    
    logger.info(f"Processing {total_entries} RSS entries")
    
    for pid, raw in rss_events_raw.items():
        try:
            # Extract source ID
            source_id = pid.split("#")[0] if "#" in pid else pid
            
            # Validate and clean raw data
            cleaned_raw = _clean_raw_data(raw, pid)
            if cleaned_raw is None:
                skipped_entries += 1
                continue
            
            # Parse JSON with multiple fallback strategies
            evs = _safe_parse_json(cleaned_raw, pid)
            if evs is None:
                skipped_entries += 1
                continue
            
            # Normalize to list format
            if not isinstance(evs, list):
                evs = [evs] if evs else []
            
            # Filter out invalid events
            valid_events = _filter_valid_events(evs, pid)
            
            if valid_events:
                feeds.setdefault(source_id, []).extend(valid_events)
                processed_entries += 1
            else:
                logger.debug(f"No valid events found in entry {pid}")
                skipped_entries += 1
                
        except Exception as e:
            logger.error(f"Unexpected error processing RSS entry {pid}: {str(e)}")
            skipped_entries += 1
            continue
    
    result = [{"rss_source_id": k, "rss_events": v} for k, v in feeds.items()]
    
    logger.info(f"RSS processing complete: {processed_entries} processed, "
                f"{skipped_entries} skipped, {len(result)} unique sources")
    
    return result


def _clean_raw_data(raw_data: Any, pid: str) -> Optional[str]:
    """
    Clean and validate raw RSS data before parsing.
    
    Args:
        raw_data: Raw data from RSS entry
        pid: Para ID for logging
        
    Returns:
        Optional[str]: Cleaned string data or None if invalid
    """
    if raw_data is None:
        logger.debug(f"RSS entry {pid} is None")
        return None
    
    # Convert to string if needed
    if not isinstance(raw_data, str):
        try:
            raw_str = str(raw_data)
        except Exception as e:
            logger.warning(f"Failed to convert RSS entry {pid} to string: {e}")
            return None
    else:
        raw_str = raw_data
    
    # Clean whitespace
    raw_str = raw_str.strip()
    
    # Check for empty content
    if not raw_str:
        logger.debug(f"RSS entry {pid} is empty after cleaning")
        return None
    
    # Check for common invalid responses
    invalid_indicators = [
        "404 not found",
        "access denied",
        "forbidden",
        "internal server error",
        "service unavailable",
        "<html>",  # HTML response instead of JSON
        "<!doctype",
    ]
    
    raw_lower = raw_str.lower()
    for indicator in invalid_indicators:
        if indicator in raw_lower:
            logger.debug(f"RSS entry {pid} contains invalid content indicator: {indicator}")
            return None
    
    return raw_str


def _safe_parse_json(raw_str: str, pid: str) -> Optional[Any]:
    """
    Safely parse JSON with multiple fallback strategies.
    
    Args:
        raw_str: Raw JSON string
        pid: Para ID for logging
        
    Returns:
        Parsed JSON data or None if all strategies fail
    """
    # Strategy 1: Use the original parse_json_any function
    try:
        result = parse_json_any(raw_str)
        logger.debug(f"Successfully parsed RSS entry {pid} with parse_json_any")
        return result
    except Exception as e:
        logger.debug(f"parse_json_any failed for {pid}: {e}")
    
    # Strategy 2: Direct JSON parsing
    try:
        result = json.loads(raw_str)
        logger.debug(f"Successfully parsed RSS entry {pid} with json.loads")
        return result
    except json.JSONDecodeError as e:
        logger.debug(f"json.loads failed for {pid}: {e}")
    
    # Strategy 3: Try to fix common JSON issues
    try:
        fixed_json = _fix_common_json_issues(raw_str)
        result = json.loads(fixed_json)
        logger.debug(f"Successfully parsed RSS entry {pid} after fixing JSON issues")
        return result
    except Exception as e:
        logger.debug(f"Fixed JSON parsing failed for {pid}: {e}")
    
    # Strategy 4: Try to extract JSON from text
    try:
        extracted_json = _extract_json_from_text(raw_str)
        if extracted_json:
            result = json.loads(extracted_json)
            logger.debug(f"Successfully extracted and parsed JSON from RSS entry {pid}")
            return result
    except Exception as e:
        logger.debug(f"JSON extraction failed for {pid}: {e}")
    
    # Log the failure with truncated content for debugging
    content_preview = raw_str[:200] + "..." if len(raw_str) > 200 else raw_str
    logger.warning(f"All JSON parsing strategies failed for RSS entry {pid}. Content preview: {content_preview}")
    
    return None


def _fix_common_json_issues(json_str: str) -> str:
    """
    Fix common JSON formatting issues.
    
    Args:
        json_str: Potentially malformed JSON string
        
    Returns:
        str: Fixed JSON string
    """
    # Remove BOM if present
    if json_str.startswith('\ufeff'):
        json_str = json_str[1:]
    
    # Fix single quotes to double quotes (basic case)
    # This is a simple fix - more complex cases might need a proper parser
    import re
    json_str = re.sub(r"'([^']*)':", r'"\1":', json_str)  # Fix keys
    json_str = re.sub(r":\s*'([^']*)'", r': "\1"', json_str)  # Fix string values
    
    # Remove trailing commas before closing brackets/braces
    json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)
    
    return json_str


def _extract_json_from_text(text: str) -> Optional[str]:
    """
    Try to extract JSON from text that might contain other content.
    
    Args:
        text: Text that might contain JSON
        
    Returns:
        Optional[str]: Extracted JSON string or None
    """
    import re
    
    # Look for JSON objects
    json_patterns = [
        r'\{.*\}',  # Object
        r'\[.*\]',  # Array
    ]
    
    for pattern in json_patterns:
        matches = re.findall(pattern, text, re.DOTALL)
        for match in matches:
            try:
                # Try to parse to validate
                json.loads(match)
                return match
            except:
                continue
    
    return None


def _filter_valid_events(events: List[Any], pid: str) -> List[Any]:
    """
    Filter out invalid or empty events.
    
    Args:
        events: List of event objects
        pid: Para ID for logging
        
    Returns:
        List[Any]: Filtered list of valid events
    """
    if not events:
        return []
    
    valid_events = []
    
    for i, event in enumerate(events):
        try:
            # Skip None or empty events
            if event is None:
                continue
            
            # Skip empty dictionaries or lists
            if isinstance(event, (dict, list)) and not event:
                continue
            
            # Skip empty strings
            if isinstance(event, str) and not event.strip():
                continue
            
            # For dictionaries, ensure they have meaningful content
            if isinstance(event, dict):
                # Skip if all values are None or empty
                if all(v is None or (isinstance(v, str) and not v.strip()) for v in event.values()):
                    continue
            
            valid_events.append(event)
            
        except Exception as e:
            logger.debug(f"Error validating event {i} in RSS entry {pid}: {e}")
            continue
    
    logger.debug(f"Filtered {len(events)} events to {len(valid_events)} valid events for {pid}")
    return valid_events


def get_rss_collection_stats(rss_collection: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Get statistics about the RSS collection for monitoring and debugging.
    
    Args:
        rss_collection: List of RSS source dictionaries
        
    Returns:
        Dict with statistics
    """
    if not rss_collection:
        return {"total_sources": 0, "total_events": 0, "sources": []}
    
    total_events = sum(len(source.get("rss_events", [])) for source in rss_collection)
    
    source_stats = []
    for source in rss_collection:
        events = source.get("rss_events", [])
        source_stats.append({
            "source_id": source.get("rss_source_id", "unknown"),
            "event_count": len(events),
            "has_events": len(events) > 0
        })
    
    return {
        "total_sources": len(rss_collection),
        "total_events": total_events,
        "sources_with_events": sum(1 for s in source_stats if s["has_events"]),
        "sources": source_stats
    }