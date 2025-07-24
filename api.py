import asyncio
import json
import logging
import os
import re
import subprocess
import sys
from contextlib import asynccontextmanager
from datetime import timedelta
from typing import Optional, Union

import uvicorn
import yaml
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel

# from fastapi.security import OAuth2PasswordRequestForm
from api_wrapper import process_full_request, set_global_model
from llm.llm_fact_checker import (
    generate_fact_check,
    generate_fact_check_with_search_grounding,
)
from rss.rss_auto_collector import (
    UPDATE_INTERVAL_HOURS,
    get_collector_status,
    manual_cleanup,
    manual_initialization,
    manual_update,
    start_background_scheduler_async,
)

# Authentication imports commented out for testing
# from auth import (
#     ACCESS_TOKEN_EXPIRE_MINUTES,
#     Token,
#     User,
#     authenticate_user,
#     create_access_token,
#     fake_users_db,
#     get_current_active_user,
# )
from simple_similarity import SentenceTransformer

# Import data persistence functions
try:
    from data_persistence import (
        auto_backup_if_needed,
        create_data_backup,
        get_data_statistics,
        restore_data_from_backup,
    )
    DATA_PERSISTENCE_AVAILABLE = True
except ImportError:
    DATA_PERSISTENCE_AVAILABLE = False

    # Note: logger not available yet, will warn later


# Configure logging with clean output
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

# Suppress noisy debug loggers
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("google").setLevel(logging.WARNING)
logging.getLogger("googleapiclient").setLevel(logging.WARNING)
logging.getLogger("asyncio").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


# Log data persistence availability
if not DATA_PERSISTENCE_AVAILABLE:
    logger.warning("Data persistence module not available - backup features disabled")

# Global model instance
_sentence_model: Optional[SentenceTransformer] = None

def get_sentence_model() -> SentenceTransformer:
    """Get the global sentence transformer model, loading it if necessary."""
    global _sentence_model
    if _sentence_model is None:
        logger.info("Loading SentenceTransformer model...")
        _sentence_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        logger.info("SentenceTransformer model loaded successfully")
    return _sentence_model

# OAuth2 scheme
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan - startup and shutdown."""
    # Startup
    logger.info("Starting up - preloading models...")
    model = get_sentence_model()  # This will load the model
    set_global_model(model)  # Set it in the api_wrapper
    
    # Check and restore data if needed (deployment persistence)
    if DATA_PERSISTENCE_AVAILABLE:
        try:
            data_stats = get_data_statistics()
            if data_stats['articles_count'] == 0 and data_stats['backup_count'] > 0:
                logger.info("🔄 No articles found but backups exist - restoring latest backup...")
                restore_success = restore_data_from_backup()
                if restore_success:
                    logger.info("✅ Data restored successfully from backup")
                    # Get updated stats after restore
                    restored_stats = get_data_statistics()
                    logger.info(f"📊 Restored: {restored_stats['articles_count']} articles, {restored_stats['vectors_count']} vectors")
                else:
                    logger.warning("❌ Failed to restore data from backup")
            else:
                logger.info(f"📊 Existing data found: {data_stats['articles_count']} articles, {data_stats['vectors_count']} vectors")
        except Exception as e:
            logger.warning(f"Error checking/restoring data: {e}")
    
    # Start RSS auto-collector in background
    await start_background_scheduler_async()
    
    # Log RSS statistics after startup
    try:
        status = get_collector_status()
        logger.info("📊 RSS System Status:")
        logger.info(f"   📰 Total articles: {status['current_articles']}")
        logger.info(f"   🔢 Total vectors: {status['current_vectors']}")
        logger.info(f"   📡 RSS feeds monitored: {status['total_feeds']}")
        logger.info(f"   ⏰ Update interval: {UPDATE_INTERVAL_HOURS} hour(s)")
        logger.info(f"   📅 Retention period: {status['retention_days']} days")
        logger.info(f"   🔄 Last update: {status['last_update'] or 'Never'}")
    except Exception as e:
        logger.warning(f"Could not get RSS status: {e}")
    
    logger.info("Startup complete - models loaded and RSS collector started")
    
    yield
    
    # Shutdown
    logger.info("Shutting down...")
    # Create backup on shutdown if available
    if DATA_PERSISTENCE_AVAILABLE:
        try:
            backup_path = create_data_backup()
            if backup_path:
                logger.info(f"📦 Created shutdown backup: {backup_path}")
        except Exception as e:
            logger.warning(f"Failed to create shutdown backup: {e}")
    logger.info("Shutdown complete")

# Create FastAPI app with lifespan
app = FastAPI(
    title="Fact Checking API",
    description="API for fact-checking text using RSS data and Wikipedia",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],  # Explicitly list allowed methods
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=3600,  # Cache preflight requests for 1 hour
)

class FactCheckRequest(BaseModel):
    text: str
    author_name: Optional[str] = None

class FactCheckRequestJSON(BaseModel):
    """Alternative request format that accepts JSON input directly."""
    input_data: Union[str, dict]

class StatusResponse(BaseModel):
    status: str
    message: str

@app.get("/health", response_model=StatusResponse)
async def health_check():
    """Health check endpoint."""
    return StatusResponse(status="healthy", message="Fact checking API is running")

@app.post("/fact-check")
async def fact_check(
    request: FactCheckRequest
):
    """
    Fact-check the provided text and return structured JSON response with emojis.
    Uses chunking for texts longer than 100 words and Google Search grounding for real-time verification.
    
    Args:
        request: FactCheckRequest containing text and optional author name
        
    Returns:
        JSONResponse with parsed analysis results including visual emojis
    """
    try:
        logger.info(f"Starting fact-check with chunking for text: {request.text[:100]}...")
        logger.info(f"Author name: {request.author_name}")
        
        # Get the global model
        model = get_sentence_model()
        
        # Create input for process_full_request
        input_data = {
            "text": request.text,
            "author": request.author_name or ""
        }
        
        # Process the request with chunking (will split if >100 words)
        result = await process_full_request(
            json.dumps(input_data, ensure_ascii=False),
            author_name=request.author_name or "",
            model=model
        )
        
        # Extract fact check results
        fact_check_results = result.get('fact_check_results', [])
        
        # Add emojis to each result for visual appeal
        for result_item in fact_check_results:
            overall_score = result_item.get('overall_accuracy_score', 0)
            
            # Add score emoji
            if overall_score >= 0.8:
                result_item['score_emoji'] = "✅"
            elif overall_score >= 0.6:
                result_item['score_emoji'] = "⚠️"
            elif overall_score >= 0.4:
                result_item['score_emoji'] = "❌"
            else:
                result_item['score_emoji'] = "🚨"
            
            # Add classification emoji
            classification = result_item.get('classification', 'unknown')
            if classification == "s":
                result_item['classification_emoji'] = "💭"
            else:
                result_item['classification_emoji'] = "📋"
        
        # Add chunking information if text was chunked
        response_content = {"analysis_results": fact_check_results}
        if result.get('is_chunked'):
            response_content.update({
                "chunking_info": {
                    "was_chunked": True,
                    "total_chunks": result.get('total_chunks', 0),
                    "original_word_count": result.get('original_word_count', 0),
                    "chunks_processed": result.get('chunking_summary', {}).get('chunks_processed', 0),
                    "chunks_failed": result.get('chunking_summary', {}).get('chunks_failed', 0)
                }
            })
        
        logger.info("Fact-check with chunking completed successfully")
        return JSONResponse(content=response_content)
        
    except Exception as e:
        logger.error(f"Unexpected error during fact-check with chunking: {e}")
        return JSONResponse(
            content={"error": f"🚨 שגיאה: {str(e)}", "analysis_results": []}, 
            status_code=500
        )

@app.post("/analyze")
async def analyze(
    request: FactCheckRequest
):
    """
    Analyze the provided text and return structured JSON response.
    
    Args:
        request: FactCheckRequest containing text and optional author name
        
    Returns:
        JSONResponse with parsed analysis results
    """
    return await fact_check(request)

@app.post("/fact-check-json", response_class=PlainTextResponse)
async def fact_check_json(
    request: FactCheckRequestJSON
):
    """
    Return the raw LLM text output for JSON input format.
    
    Args:
        request: FactCheckRequestJSON containing input_data as JSON or string
        
    Returns:
        PlainTextResponse with the raw LLM text output
    """
    try:
        logger.info(f"Starting JSON fact-check...")
        
        # Handle both dict and string input
        if isinstance(request.input_data, dict):
            text = request.input_data.get("text", "")
            author_name = request.input_data.get("author", "")
        else:
            # If it's a string, treat it as the text
            text = request.input_data
            author_name = ""
        
        logger.info(f"Extracted text: {text[:100]}...")
        logger.info(f"Extracted author: {author_name}")
        
        # Call the fact checker directly with minimal parameters
        result = generate_fact_check(
            user_event_id=1,
            user_event_quote=text,
            rss_quotes="[]",
            wiki_quotes="[]",
            correlation_json="{}",
            author=author_name
        )
        
        logger.info("JSON fact-check completed successfully")
        return PlainTextResponse(content=result, media_type="text/plain; charset=utf-8")
        
    except Exception as e:
        logger.error(f"Unexpected error during JSON fact-check: {e}")
        return PlainTextResponse(content=f"שגיאה: {str(e)}", status_code=500, media_type="text/plain; charset=utf-8")

@app.post("/update-vectors")
async def update_vectors():
    """
    Update RSS vectors by running the RSS auto-collector.
    """
    try:
        logger.info("Starting RSS data update")
        
        # Use the new RSS auto-collector
        await manual_update()
        
        logger.info("RSS update completed successfully")
        return {"status": "success", "message": "RSS data updated successfully"}
        
    except Exception as e:
        logger.error(f"RSS update error: {e}")
        return {"status": "error", "message": f"RSS update error: {str(e)}"}

@app.get("/rss-status")
async def rss_status():
    """
    Get the current status of the RSS auto-collector.
    """
    try:
        status = get_collector_status()
        return {"status": "success", "data": status}
    except Exception as e:
        logger.error(f"Error getting RSS status: {e}")
        return {"status": "error", "message": f"Error getting RSS status: {str(e)}"}

@app.post("/initialize-rss")
async def initialize_rss():
    """
    Initialize RSS vectors for existing articles that don't have vectors yet.
    """
    try:
        logger.info("Starting RSS initialization with existing data")
        
        await manual_initialization()
        
        logger.info("RSS initialization completed successfully")
        return {"status": "success", "message": "RSS initialization completed successfully"}
        
    except Exception as e:
        logger.error(f"RSS initialization error: {e}")
        return {"status": "error", "message": f"RSS initialization error: {str(e)}"}

@app.post("/cleanup-rss")
async def cleanup_rss():
    """Manually trigger RSS cleanup."""
    try:
        logger.info("Manual RSS cleanup triggered")
        await manual_cleanup()
        return {
            "status": "success",
            "message": "RSS cleanup completed successfully"
        }
    except Exception as e:
        logger.error(f"Error during manual RSS cleanup: {e}")
        raise HTTPException(status_code=500, detail=f"RSS cleanup failed: {str(e)}")

@app.options("/fact-check")
async def fact_check_options():
    """Handle OPTIONS request for fact-check endpoint."""
    return {"status": "ok"}

@app.options("/analyze")
async def analyze_options():
    """Handle OPTIONS request for analyze endpoint."""
    return {"status": "ok"}

@app.options("/fact-check-json")
async def fact_check_json_options():
    """Handle OPTIONS request for fact-check-json endpoint."""
    return {"status": "ok"}

@app.options("/update-vectors")
async def update_vectors_options():
    """Handle OPTIONS request for update-vectors endpoint."""
    return {"status": "ok"}

@app.options("/initialize-rss")
async def initialize_rss_options():
    """Handle OPTIONS request for initialize-rss endpoint."""
    return {"status": "ok"}

@app.options("/cleanup-rss")
async def cleanup_rss_options():
    """Handle OPTIONS request for cleanup-rss endpoint."""
    return {"status": "ok"}

@app.post("/fact-check-raw", response_class=PlainTextResponse)
async def fact_check_raw(
    request: FactCheckRequest
):
    """
    Return the raw LLM text output for the provided text.
    
    Args:
        request: FactCheckRequest containing text and optional author name
        
    Returns:
        PlainTextResponse with the raw LLM text output
    """
    try:
        logger.info(f"Starting raw fact-check for text: {request.text[:100]}...")
        logger.info(f"Author name: {request.author_name}")
        
        # Call the fact checker directly with minimal parameters
        result = generate_fact_check(
            user_event_id=1,
            user_event_quote=request.text,
            rss_quotes="[]",
            wiki_quotes="[]",
            correlation_json="{}",
            author=request.author_name or ""
        )
        
        logger.info("Raw fact-check completed successfully")
        return PlainTextResponse(content=result, media_type="text/plain; charset=utf-8")
        
    except Exception as e:
        logger.error(f"Unexpected error during raw fact-check: {e}")
        return PlainTextResponse(content=f"🚨 שגיאה: {str(e)}", status_code=500, media_type="text/plain; charset=utf-8")

@app.options("/fact-check-raw")
async def fact_check_raw_options():
    """Handle OPTIONS request for fact-check-raw endpoint."""
    return {"status": "ok"}

@app.post("/backup-data")
async def backup_data():
    """Create a backup of RSS data."""
    if not DATA_PERSISTENCE_AVAILABLE:
        raise HTTPException(status_code=503, detail="Data persistence module not available")
    
    try:
        logger.info("Manual data backup triggered")
        backup_path = create_data_backup()
        if backup_path:
            return {
                "status": "success",
                "message": "Data backup created successfully",
                "backup_path": backup_path
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to create backup")
    except Exception as e:
        logger.error(f"Error creating backup: {e}")
        raise HTTPException(status_code=500, detail=f"Backup failed: {str(e)}")

@app.post("/restore-data")
async def restore_data(backup_path: Optional[str] = None):
    """Restore RSS data from backup."""
    if not DATA_PERSISTENCE_AVAILABLE:
        raise HTTPException(status_code=503, detail="Data persistence module not available")
    
    try:
        logger.info(f"Manual data restore triggered: {backup_path or 'latest backup'}")
        success = restore_data_from_backup(backup_path)
        if success:
            return {
                "status": "success",
                "message": "Data restored successfully"
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to restore data")
    except Exception as e:
        logger.error(f"Error restoring data: {e}")
        raise HTTPException(status_code=500, detail=f"Restore failed: {str(e)}")

@app.get("/data-stats")
async def data_stats():
    """Get comprehensive data statistics including backups."""
    try:
        # Get RSS collector status
        rss_status = get_collector_status()
        
        # Get data persistence stats if available
        persistence_stats = {}
        if DATA_PERSISTENCE_AVAILABLE:
            persistence_stats = get_data_statistics()
        
        return {
            "status": "success",
            "rss_collector": rss_status,
            "data_persistence": persistence_stats,
            "persistence_available": DATA_PERSISTENCE_AVAILABLE
        }
    except Exception as e:
        logger.error(f"Error getting data stats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")

def parse_yaml_response(text_response: str) -> list:
    """
    Parse the response from the LLM into a structured format.
    Handles both JSON and YAML formats, with special handling for Hebrew text.
    """
    try:
        # Clean the response - remove markdown code block markers
        cleaned_response = text_response.strip()
        
        # Remove ```json, ```yaml and ``` markers if present
        if cleaned_response.startswith('```json'):
            cleaned_response = cleaned_response[7:]  # Remove ```json
        elif cleaned_response.startswith('```yaml'):
            cleaned_response = cleaned_response[7:]  # Remove ```yaml
        elif cleaned_response.startswith('```'):
            cleaned_response = cleaned_response[3:]   # Remove ```
            
        if cleaned_response.endswith('```'):
            cleaned_response = cleaned_response[:-3]  # Remove trailing ```
            
        cleaned_response = cleaned_response.strip()
        
        # Try to parse as JSON first (silently, no warnings yet)
        try:
            parsed_data = json.loads(cleaned_response)
            if isinstance(parsed_data, list):
                logger.info(f"Successfully parsed JSON array with {len(parsed_data)} items")
                return parsed_data
            elif isinstance(parsed_data, dict):
                logger.info("Successfully parsed JSON object, wrapping in array")
                return [parsed_data]  # Wrap single object in list
        except json.JSONDecodeError:
            # Don't log warning yet - try other methods first
            pass
        
        # If JSON parsing fails, try to extract JSON array from the response
        json_match = re.search(r'\[.*\]', cleaned_response, re.DOTALL)
        if json_match:
            try:
                json_str = json_match.group(0)
                parsed_data = json.loads(json_str)
                if isinstance(parsed_data, list):
                    logger.info(f"Successfully extracted and parsed JSON array with {len(parsed_data)} items")
                    return parsed_data
            except json.JSONDecodeError:
                # Still don't log warning - continue trying
                pass
        
        # Try to extract individual JSON objects and combine them
        json_objects = re.findall(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', cleaned_response, re.DOTALL)
        if json_objects:
            parsed_objects = []
            for obj_str in json_objects:
                try:
                    obj = json.loads(obj_str)
                    parsed_objects.append(obj)
                except json.JSONDecodeError:
                    continue
            if parsed_objects:
                logger.info(f"Successfully extracted {len(parsed_objects)} JSON objects")
                return parsed_objects
        
        # If JSON extraction fails, try YAML with better Unicode handling
        try:
            # Use safe_load with explicit UTF-8 encoding handling
            parsed_data = yaml.safe_load(cleaned_response)
            if isinstance(parsed_data, list):
                logger.info(f"Successfully parsed YAML array with {len(parsed_data)} items")
                return parsed_data
            elif isinstance(parsed_data, dict):
                logger.info("Successfully parsed YAML object, wrapping in array")
                return [parsed_data]  # Wrap single object in list
        except yaml.YAMLError:
            # Still don't log warning - continue trying
            pass
        
        # Only NOW log a warning since all structured parsing attempts failed
        logger.warning("All structured parsing attempts failed, attempting text analysis...")
        
        # Check if the response contains any useful information
        if len(cleaned_response) > 50:
            # Try to extract key information manually
            basic_response = {
                "sentence_text": cleaned_response[:200] + "..." if len(cleaned_response) > 200 else cleaned_response,
                "classification": "not_s",
                "sentence_nature": "Direct Assertion by user_event_quote",
                "objectivity_score": 0.5,
                "event_happened_score": 0.5,
                "quote_correctness_score": None,
                "overall_accuracy_score": 0.5,
                "reasoning": "לא ניתן היה לנתח את התגובה במבנה JSON תקין. התגובה הגולמית זמינה בשדה sentence_text.",
                "sources": [],
                "search_queries_used": [],
                "low_score_explanation": "בעיה בפורמט התגובה מהמודל"
            }
            logger.info("Created basic response from unparseable text")
            return [basic_response]
        
        # If all parsing attempts fail, return a default structure
        logger.error(f"Could not parse response, returning empty list. Response preview: {text_response[:200]}...")
        return []
        
    except Exception as e:
        logger.error(f"Error in parse_yaml_response: {e}")
        logger.error(f"Response that caused error: {text_response[:500]}...")
        return []

def _preprocess_yaml_for_hebrew(yaml_content: str) -> str:
    """
    Preprocess YAML content to handle Hebrew text properly.
    This function quotes Hebrew text values to prevent YAML parsing errors.
    """
    lines = yaml_content.split('\n')
    processed_lines = []
    
    for line in lines:
        # Check if line contains Hebrew text after a colon
        if ':' in line and any('\u0590' <= char <= '\u05FF' for char in line):
            # Split on the first colon
            parts = line.split(':', 1)
            if len(parts) == 2:
                key_part = parts[0]
                value_part = parts[1].strip()
                
                # If value contains Hebrew and isn't already quoted
                if value_part and any('\u0590' <= char <= '\u05FF' for char in value_part):
                    if not (value_part.startswith('"') and value_part.endswith('"')):
                        # Quote the value
                        value_part = f'"{value_part}"'
                    line = f"{key_part}: {value_part}"
        
        processed_lines.append(line)
    
    return '\n'.join(processed_lines)


@app.post("/backup-data")
async def backup_data():
    """Create a backup of RSS data."""
    if not DATA_PERSISTENCE_AVAILABLE:
        raise HTTPException(status_code=503, detail="Data persistence module not available")
    
    try:
        logger.info("Manual data backup triggered")
        backup_path = create_data_backup()
        if backup_path:
            return {
                "status": "success",
                "message": "Data backup created successfully",
                "backup_path": backup_path
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to create backup")
    except Exception as e:
        logger.error(f"Error creating backup: {e}")
        raise HTTPException(status_code=500, detail=f"Backup failed: {str(e)}")

@app.post("/restore-data")
async def restore_data(backup_path: Optional[str] = None):
    """Restore RSS data from backup."""
    if not DATA_PERSISTENCE_AVAILABLE:
        raise HTTPException(status_code=503, detail="Data persistence module not available")
    
    try:
        logger.info(f"Manual data restore triggered: {backup_path or 'latest backup'}")
        success = restore_data_from_backup(backup_path)
        if success:
            return {
                "status": "success",
                "message": "Data restored successfully"
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to restore data")
    except Exception as e:
        logger.error(f"Error restoring data: {e}")
        raise HTTPException(status_code=500, detail=f"Restore failed: {str(e)}")

@app.get("/data-stats")
async def data_stats():
    """Get comprehensive data statistics including backups."""
    try:
        # Get RSS collector status
        rss_status = get_collector_status()
        
        # Get data persistence stats if available
        persistence_stats = {}
        if DATA_PERSISTENCE_AVAILABLE:
            persistence_stats = get_data_statistics()
        
        return {
            "status": "success",
            "rss_collector": rss_status,
            "data_persistence": persistence_stats,
            "persistence_available": DATA_PERSISTENCE_AVAILABLE
        }
    except Exception as e:
        logger.error(f"Error getting data stats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)