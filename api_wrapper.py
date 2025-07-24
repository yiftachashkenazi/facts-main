#!/usr/bin/env python3

import asyncio
import json
import logging
from typing import Any, Dict, Optional

from llm.llm_fact_checker import generate_fact_check_with_search_grounding
from main1 import run_calibration_phase, run_full_analysis
from simple_similarity import SentenceTransformer
from utils.text_chunker import (
    chunk_text_by_words,
    count_words,
    prepare_chunk_with_author,
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global model reference - will be set by the API module
_global_model: Optional[SentenceTransformer] = None

def set_global_model(model: SentenceTransformer):
    """Set the global model instance to be used by the wrapper."""
    global _global_model
    _global_model = model
    logger.info("Global model set in api_wrapper")

def get_global_model() -> Optional[SentenceTransformer]:
    """Get the global model instance."""
    return _global_model

async def process_calibration_request(input_data: str) -> Dict[str, Any]:
    """
    Process calibration phase only.
    
    Args:
        input_data: JSON string or plain text input
        
    Returns:
        Dictionary containing calibration results
    """
    try:
        logger.info("Starting calibration phase processing")
        model = get_global_model()
        result = await run_calibration_phase(input_data, model=model)
        logger.info("Calibration phase completed successfully")
        return {
            "status": "success",
            "phase": "calibration",
            "data": result
        }
    except Exception as e:
        logger.error(f"Calibration phase failed: {e}")
        return {
            "status": "error",
            "phase": "calibration",
            "error": str(e)
        }


async def process_full_request(
    query: str, 
    author_name: str = "", 
    model: Optional[SentenceTransformer] = None
) -> Dict[str, Any]:
    """
    Process a full fact-checking request with enhanced search grounding and text chunking.
    For texts longer than 250 words, splits into chunks and processes each separately.
    """
    logger.info(f"Processing full request: {query[:100]}...")
    
    try:
        # Use the global model if none provided
        if model is None:
            model = get_global_model()
            if model is None:
                raise ValueError("No model available - global model not set")
        
        # Parse JSON if needed to extract text and author
        original_query = query
        try:
            json_input = json.loads(query)
            if isinstance(json_input, dict):
                text_content = json_input.get("text", json_input.get("content", json_input.get("query", query)))
                if not author_name and "author" in json_input:
                    author_name = json_input["author"]
            else:
                text_content = query
        except json.JSONDecodeError:
            text_content = query
        
        # Count words in the text content (excluding author)
        word_count = count_words(text_content)
        logger.info(f"Text word count: {word_count}")
        
        # Check if text needs chunking (>100 words)
        if word_count > 100:
            logger.info(f"Text exceeds 100 words ({word_count}), splitting into chunks...")
            return await _process_chunked_text(text_content, author_name, model)
        else:
            logger.info("Text is within word limit, processing normally...")
            # Process normally for short text
            result = await run_full_analysis(original_query, model)
            
            # Check if we have sufficient sources from RSS and Wikipedia
            rss_sources = result.get('rss_payload', [])
            wiki_sources = result.get('character_info', {})
            
            # Determine if we need enhanced search grounding
            needs_search_grounding = (
                len(rss_sources) < 3 or  # Less than 3 RSS sources
                not wiki_sources or      # No Wikipedia information
                not any(source.get('text', '').strip() for source in rss_sources)  # Empty RSS content
            )
            
            if needs_search_grounding:
                logger.info("Insufficient sources detected - using enhanced search grounding")
                
                # Use search grounding for fact-checking
                enhanced_result = generate_fact_check_with_search_grounding(
                    user_event_id=1,
                    user_event_quote=text_content,
                    rss_quotes=json.dumps(rss_sources),
                    wiki_quotes=json.dumps(wiki_sources),
                    correlation_json=json.dumps(result.get('matches_dict', {})),
                    query=text_content,
                    author=author_name,
                    author_info=json.dumps(result.get('author_info', {})),
                    user_events=json.dumps(result.get('user_events', [])),
                    rss_payload=json.dumps(rss_sources),
                    matches_dict=json.dumps(result.get('matches_dict', {})),
                    character_info=json.dumps(wiki_sources),
                    calibration_data=json.dumps(result.get('calibration_data', {}))
                )
                
                # Parse the enhanced result and add to fact_check_results
                try:
                    # Parse the YAML response into JSON
                    from api import parse_yaml_response
                    parsed_enhanced_result = parse_yaml_response(enhanced_result)
                    result['fact_check_results'] = parsed_enhanced_result
                except Exception as parse_error:
                    logger.error(f"Error parsing enhanced result: {parse_error}")
                    # Fallback: keep original fact_check_results
                    pass
                
                result['search_grounding_used'] = True
                result['search_grounding_reason'] = "Insufficient RSS/Wikipedia sources"
            else:
                logger.info("Sufficient sources available - using standard fact-checking")
                result['search_grounding_used'] = False
            
            return result
        
    except Exception as e:
        logger.error(f"Error in process_full_request: {e}")
        return {
            "error": str(e),
            "search_grounding_used": False,
            "query": query,
            "author": author_name
        }


async def _process_chunked_text(text_content: str, author_name: str, model: SentenceTransformer) -> Dict[str, Any]:
    """
    Process text by splitting it into chunks and processing each chunk separately.
    """
    # Split text into chunks (smaller chunks for better processing)
    chunks = chunk_text_by_words(text_content, target_words=100, max_chunk_words=150)
    total_chunks = len(chunks)
    
    logger.info(f"Split text into {total_chunks} chunks")
    
    # Process each chunk
    chunk_results = []
    combined_fact_check_results = []
    combined_user_events = []
    combined_rss_payload = []
    combined_matches_dict = {}
    combined_character_info = {}
    combined_calibration_data = []
    
    # Track combined statistics
    total_rss_matches = 0
    total_wikipedia_characters = set()
    total_events_analyzed = 0
    total_fact_checks_completed = 0
    
    for i, (chunk_text, chunk_word_count) in enumerate(chunks, 1):
        logger.info(f"Processing chunk {i}/{total_chunks} ({chunk_word_count} words)")
        
        # Prepare chunk with author info and chunk metadata
        formatted_chunk = prepare_chunk_with_author(
            chunk_text, author_name, chunk_index=i, total_chunks=total_chunks
        )
        
        try:
            # Run full analysis on this chunk
            chunk_result = await run_full_analysis(formatted_chunk, model)
            
            # Store the chunk result
            chunk_results.append({
                "chunk_index": i,
                "chunk_text": chunk_text,
                "word_count": chunk_word_count,
                "result": chunk_result
            })
            
            # Combine results from this chunk
            if isinstance(chunk_result, dict):
                # Check if we need enhanced search grounding for this chunk
                rss_sources = chunk_result.get('rss_payload', [])
                wiki_sources = chunk_result.get('character_info', {})
                
                needs_search_grounding = (
                    len(rss_sources) < 3 or  # Less than 3 RSS sources
                    not wiki_sources or      # No Wikipedia information
                    not any(source.get('text', '').strip() for source in rss_sources)  # Empty RSS content
                )
                
                if needs_search_grounding:
                    logger.info(f"Chunk {i}: Insufficient sources detected - using enhanced search grounding")
                    
                    # Use search grounding for fact-checking this chunk
                    enhanced_result = generate_fact_check_with_search_grounding(
                        user_event_id=1,
                        user_event_quote=chunk_text,
                        rss_quotes=json.dumps(rss_sources),
                        wiki_quotes=json.dumps(wiki_sources),
                        correlation_json=json.dumps(chunk_result.get('matches_dict', {})),
                        query=chunk_text,
                        author=author_name,
                        author_info=json.dumps(chunk_result.get('author_info', {})),
                        user_events=json.dumps(chunk_result.get('user_events', [])),
                        rss_payload=json.dumps(rss_sources),
                        matches_dict=json.dumps(chunk_result.get('matches_dict', {})),
                        character_info=json.dumps(wiki_sources),
                        calibration_data=json.dumps(chunk_result.get('calibration_data', {}))
                    )
                    
                    # Parse the enhanced result and add to fact_check_results
                    try:
                        from api import parse_yaml_response
                        parsed_enhanced_result = parse_yaml_response(enhanced_result)
                        chunk_fact_checks = parsed_enhanced_result
                    except Exception as parse_error:
                        logger.error(f"Error parsing enhanced result for chunk {i}: {parse_error}")
                        chunk_fact_checks = chunk_result.get('fact_check_results', [])
                else:
                    logger.info(f"Chunk {i}: Sufficient sources available - using standard fact-checking")
                    chunk_fact_checks = chunk_result.get('fact_check_results', [])
                
                # Add chunk index to fact check results
                for fact_check in chunk_fact_checks:
                    if isinstance(fact_check, dict):
                        fact_check['chunk_index'] = i
                        fact_check['chunk_text_preview'] = chunk_text[:100] + "..." if len(chunk_text) > 100 else chunk_text
                combined_fact_check_results.extend(chunk_fact_checks)
                
                # Combine other data
                combined_user_events.extend(chunk_result.get('user_events', []))
                combined_rss_payload.extend(chunk_result.get('rss_payload', []))
                
                # Merge dictionaries
                chunk_matches = chunk_result.get('matches_dict', {})
                for key, value in chunk_matches.items():
                    combined_matches_dict[f"chunk_{i}_{key}"] = value
                
                chunk_characters = chunk_result.get('character_info', {})
                for key, value in chunk_characters.items():
                    combined_character_info[f"chunk_{i}_{key}"] = value
                
                combined_calibration_data.extend(chunk_result.get('calibration_data', []))
                
                # Update statistics
                total_rss_matches += len([m for m in chunk_matches.values() if m.get("Match")])
                total_wikipedia_characters.update(chunk_characters.keys())
                total_events_analyzed += len(chunk_result.get('user_events', []))
                total_fact_checks_completed += len(chunk_fact_checks)
            
            logger.info(f"Chunk {i} processed successfully")
            
        except Exception as e:
            logger.error(f"Error processing chunk {i}: {e}")
            chunk_results.append({
                "chunk_index": i,
                "chunk_text": chunk_text,
                "word_count": chunk_word_count,
                "error": str(e)
            })
    
    # Create combined final result
    combined_result = {
        "status": "success",
        "query": text_content,
        "author": author_name,
        "is_chunked": True,
        "total_chunks": total_chunks,
        "original_word_count": count_words(text_content),
        "chunking_summary": {
            "chunks_processed": len([r for r in chunk_results if "error" not in r]),
            "chunks_failed": len([r for r in chunk_results if "error" in r]),
            "total_word_count": sum(chunk[1] for chunk in chunks)
        },
        "summary": {
            "total_events_analyzed": total_events_analyzed,
            "fact_checks_completed": total_fact_checks_completed,
            "rss_matches": total_rss_matches,
            "wikipedia_characters": len(total_wikipedia_characters)
        },
        "fact_check_results": combined_fact_check_results,
        "user_events": combined_user_events,
        "rss_payload": combined_rss_payload,
        "matches_dict": combined_matches_dict,
        "character_info": combined_character_info,
        "calibration_data": combined_calibration_data,
        "chunk_details": chunk_results,
        "search_grounding_used": any(
            chunk_result.get("result", {}).get("search_grounding_used", False) 
            for chunk_result in chunk_results if "error" not in chunk_result and "result" in chunk_result
        )
    }
    
    logger.info(f"Chunked processing complete: {total_chunks} chunks, {total_fact_checks_completed} total fact checks")
    
    return combined_result


def run_calibration_sync(input_data: str) -> Dict[str, Any]:
    """Synchronous wrapper for calibration phase."""
    return asyncio.run(process_calibration_request(input_data))


def run_full_sync(input_data: str) -> Dict[str, Any]:
    """Synchronous wrapper for full analysis."""
    return asyncio.run(process_full_request(input_data))


# Example usage functions
async def example_json_input():
    """Example of how to use with JSON input containing author."""
    json_input = {
        "author": "John Doe",
        "text": "This is a sample text to analyze for fact-checking."
    }
    
    # Phase 1: Get calibration data
    calibration_result = await process_calibration_request(json.dumps(json_input))
    print("Calibration Result:", json.dumps(calibration_result, indent=2, ensure_ascii=False))
    
    # Phase 2: Get full analysis
    full_result = await process_full_request(json.dumps(json_input))
    print("Full Result:", json.dumps(full_result, indent=2, ensure_ascii=False))


async def example_text_input():
    """Example of how to use with plain text input."""
    text_input = "This is a sample text to analyze for fact-checking."
    
    # Phase 1: Get calibration data
    calibration_result = await process_calibration_request(text_input)
    print("Calibration Result:", json.dumps(calibration_result, indent=2, ensure_ascii=False))
    
    # Phase 2: Get full analysis
    full_result = await process_full_request(text_input)
    print("Full Result:", json.dumps(full_result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    # Example usage
    print("Running example with JSON input...")
    asyncio.run(example_json_input())
    
    print("\nRunning example with text input...")
    asyncio.run(example_text_input()) 