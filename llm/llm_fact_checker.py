#!/usr/bin/env python3
"""
Enhanced Fact Checker with Hebrew language detection and Hebrew-first processing
"""

import base64
import json
import logging
import os
import re
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from google import genai
from google.genai.types import (
    GenerateContentConfig,
    GenerationConfig,
    GoogleSearch,
    HarmBlockThreshold,
    HarmCategory,
    SafetySetting,
    Tool,
)

# Import the RSS TF-IDF matcher
try:
    from rss_tfidf_matcher import find_relevant_rss_articles
except ImportError:
    # Fallback if the module is not available
    def find_relevant_rss_articles(claim_text: str, max_articles: int = 5) -> str:
        return "RSS TF-IDF matcher not available."

logger = logging.getLogger(__name__)

def detect_language(text: str) -> Tuple[str, float]:
    """
    Detect if text is primarily in Hebrew or another language.
    Returns: (language_code, confidence_score)
    """
    # Count Hebrew characters (excluding punctuation and numbers)
    hebrew_chars = 0
    latin_chars = 0
    total_chars = 0
    
    for char in text:
        if char.isalpha():
            total_chars += 1
            if '\u0590' <= char <= '\u05FF':  # Hebrew Unicode range
                hebrew_chars += 1
            elif 'a' <= char.lower() <= 'z':  # Latin characters
                latin_chars += 1
    
    if total_chars == 0:
        return "unknown", 0.0
    
    hebrew_ratio = hebrew_chars / total_chars
    latin_ratio = latin_chars / total_chars
    
    # Determine primary language with improved mixed language detection
    if hebrew_ratio > 0.6:  # If more than 60% Hebrew characters
        return "hebrew", hebrew_ratio
    elif latin_ratio > 0.6:  # If more than 60% Latin characters
        return "english", latin_ratio
    elif hebrew_ratio > 0.1 and latin_ratio > 0.1:  # Both languages present
        return "mixed", max(hebrew_ratio, latin_ratio)
    elif hebrew_ratio > 0.3:  # Lower threshold for Hebrew detection
        return "hebrew", hebrew_ratio
    elif latin_ratio > 0.3:  # Lower threshold for English detection
        return "english", latin_ratio
    else:
        return "mixed", max(hebrew_ratio, latin_ratio)

def split_sentences(text: str) -> List[str]:
    """
    Split text into sentences, handling Hebrew and English properly.
    Returns list of non-empty sentences with their positions.
    """
    # Simple sentence splitting for Hebrew and English
    # Split on periods, exclamation marks, question marks
    import re
    
    # Replace multiple spaces with single space and clean up
    text = re.sub(r'\s+', ' ', text.strip())
    
    # Split on sentence endings, but keep the punctuation
    sentences = re.split(r'([.!?])', text)
    
    # Reconstruct sentences with their punctuation
    result_sentences = []
    current_sentence = ""
    
    for i, part in enumerate(sentences):
        if part in '.!?':
            current_sentence += part
            if current_sentence.strip():
                result_sentences.append(current_sentence.strip())
            current_sentence = ""
        else:
            current_sentence += part
    
    # Add any remaining text as a sentence
    if current_sentence.strip():
        result_sentences.append(current_sentence.strip())
    
    # Filter out very short sentences (less than 5 characters)
    result_sentences = [s for s in result_sentences if len(s.strip()) >= 5]
    
    return result_sentences

def retry_api_call(func, max_retries: int = 3, base_delay: float = 1.0):
    """
    Retry API calls with exponential backoff for handling temporary API issues.
    """
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            error_str = str(e)
            
            # Check if it's a retryable error (503, rate limiting, etc.)
            if any(code in error_str for code in ['503', 'UNAVAILABLE', 'overloaded', 'rate limit', 'quota']):
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)  # Exponential backoff
                    logger.warning(f"API call failed (attempt {attempt + 1}/{max_retries}): {error_str}. Retrying in {delay} seconds...")
                    time.sleep(delay)
                    continue
                else:
                    logger.error(f"API call failed after {max_retries} attempts: {error_str}")
                    raise
            else:
                # Non-retryable error, raise immediately
                logger.error(f"Non-retryable API error: {error_str}")
                raise
    
    return None

def _create_fallback_analysis_for_sentence(sentence: str, sentence_number: int, error_reason: str = "", language: str = "hebrew") -> Dict[str, Any]:
    """
    Create a complete fallback analysis for a single sentence with language-appropriate text.
    """
    if language == "hebrew":
        return {
            "sentence_text": sentence,
            "sentence_number": sentence_number,
            "classification": "not_s",
            "sentence_nature": "טענה ישירה מהמשתמש",
            "main_event_identified": "ניתוח נכשל",
            "arguments_extracted": [],
            "objectivity_score": 0.5,
            "event_happened_score": 0.0,
            "quote_correctness_score": None,
            "overall_accuracy_score": 0.0,
            "similarity_score": 0.0,
            "hidden_interpretation_detected": False,
            "interpretation_details": None,
            "reasoning": error_reason or "לא ניתן היה לנתח את המשפט",
            "sources": [],
            "search_queries_used": [],
            "low_score_explanation": "המודל לא הצליח לנתח את המשפט"
        }
    else:  # English fallback
        return {
            "sentence_text": sentence,
            "sentence_number": sentence_number,
            "classification": "not_s",
            "sentence_nature": "Direct Assertion by user_event_quote",
            "main_event_identified": "Analysis failed",
            "arguments_extracted": [],
            "objectivity_score": 0.5,
            "event_happened_score": 0.0,
            "quote_correctness_score": None,
            "overall_accuracy_score": 0.0,
            "similarity_score": 0.0,
            "hidden_interpretation_detected": False,
            "interpretation_details": None,
            "reasoning": error_reason or "Could not analyze the sentence",
            "sources": [],
            "search_queries_used": [],
            "low_score_explanation": "The model failed to analyze the sentence"
        }

def _create_complete_fallback_analysis(sentences: List[str], error_reason: str = "", language: str = "hebrew") -> List[Dict[str, Any]]:
    """
    Create complete fallback analysis for all sentences with language-appropriate text.
    """
    return [
        _create_fallback_analysis_for_sentence(sentence, i + 1, error_reason, language)
        for i, sentence in enumerate(sentences)
    ]

def _parse_and_validate_llm_response(raw_response: str, input_sentences: List[str], language: str = "hebrew") -> List[Dict[str, Any]]:
    """
    Parse and validate LLM response, ensuring complete coverage of all input sentences.
    This is the main validation pipeline that runs AFTER the LLM completes its work.
    """
    logger.info(f"Parsing LLM response of {len(raw_response)} characters for {len(input_sentences)} sentences in {language}")
    
    # Step 1: Clean the response text
    cleaned_response = raw_response.strip()
    
    # Remove markdown code block markers if present
    if cleaned_response.startswith('```json'):
        cleaned_response = cleaned_response[7:]
    elif cleaned_response.startswith('```'):
        cleaned_response = cleaned_response[3:]
        
    if cleaned_response.endswith('```'):
        cleaned_response = cleaned_response[:-3]
        
    cleaned_response = cleaned_response.strip()
    
    # Step 2: Try to extract JSON from the cleaned response
    try:
        # Look for JSON array first
        json_array_match = re.search(r'\[\s*\{.*?\}\s*\]', cleaned_response, re.DOTALL)
        if json_array_match:
            json_candidate = json_array_match.group(0)
            try:
                parsed_analysis = json.loads(json_candidate)
                logger.info(f"Successfully extracted JSON array with {len(parsed_analysis)} objects")
            except json.JSONDecodeError:
                logger.warning("JSON array extraction failed, trying full text parse")
                parsed_analysis = json.loads(cleaned_response)
        else:
            # Try parsing the full cleaned response
            parsed_analysis = json.loads(cleaned_response)
        
        # Ensure it's a list
        if isinstance(parsed_analysis, dict):
            parsed_analysis = [parsed_analysis]
        elif not isinstance(parsed_analysis, list):
            logger.warning("Response is not a list or dict, creating fallback")
            error_msg = "התגובה אינה במבנה רשימה תקין" if language == "hebrew" else "Response is not in valid list format"
            return _create_complete_fallback_analysis(input_sentences, error_msg, language)
        
    except json.JSONDecodeError as e:
        logger.warning(f"JSON parsing completely failed: {e}")
        error_msg = "התגובה אינה בפורמט JSON תקין" if language == "hebrew" else "Response is not in valid JSON format"
        return _create_complete_fallback_analysis(input_sentences, error_msg, language)
    
    # Step 3: Validate each analysis object has required fields
    validated_analysis = []
    for i, analysis in enumerate(parsed_analysis):
        if not isinstance(analysis, dict):
            logger.warning(f"Analysis {i} is not a dict, skipping")
            continue
            
        # Language-appropriate defaults
        if language == "hebrew":
            defaults = {
                "sentence_text": f"משפט {i+1}",
                "sentence_nature": "טענה ישירה מהמשתמש",
                "main_event_identified": "",
                "reasoning": "ניתוח בסיסי",
                "low_score_explanation": ""
            }
        else:
            defaults = {
                "sentence_text": f"Sentence {i+1}",
                "sentence_nature": "Direct Assertion by user_event_quote",
                "main_event_identified": "",
                "reasoning": "Basic analysis",
                "low_score_explanation": ""
            }
            
        # Ensure all required fields exist with proper defaults
        validated_obj = {
            "sentence_text": analysis.get("sentence_text", defaults["sentence_text"]),
            "sentence_number": analysis.get("sentence_number", i+1),
            "classification": analysis.get("classification", "not_s"),
            "sentence_nature": analysis.get("sentence_nature", defaults["sentence_nature"]),
            "main_event_identified": analysis.get("main_event_identified", defaults["main_event_identified"]),
            "arguments_extracted": analysis.get("arguments_extracted", []),
            "objectivity_score": float(analysis.get("objectivity_score", 0.5)),
            "event_happened_score": float(analysis.get("event_happened_score", 0.5)),
            "quote_correctness_score": analysis.get("quote_correctness_score"),
            "overall_accuracy_score": float(analysis.get("overall_accuracy_score", 0.5)),
            "similarity_score": float(analysis.get("similarity_score", 0.5)),
            "hidden_interpretation_detected": bool(analysis.get("hidden_interpretation_detected", False)),
            "interpretation_details": analysis.get("interpretation_details"),
            "reasoning": analysis.get("reasoning", defaults["reasoning"]),
            "sources": analysis.get("sources", []),
            "search_queries_used": analysis.get("search_queries_used", []),
            "low_score_explanation": analysis.get("low_score_explanation", defaults["low_score_explanation"])
        }
        validated_analysis.append(validated_obj)
    
    # Step 4: Check sentence coverage and add missing sentences
    analyzed_sentences = {obj["sentence_text"].strip().lower() for obj in validated_analysis}
    missing_sentences = []
    
    for i, input_sentence in enumerate(input_sentences):
        sentence_clean = input_sentence.strip().lower()
        
        # Check if this sentence is covered
        found_coverage = False
        for analyzed_sentence in analyzed_sentences:
            # Exact match or high overlap
            if sentence_clean == analyzed_sentence:
                found_coverage = True
                break
            # Check for high word overlap (80%+)
            input_words = set(sentence_clean.split())
            analyzed_words = set(analyzed_sentence.split())
            if len(input_words) > 0:
                overlap = len(input_words.intersection(analyzed_words))
                if overlap / len(input_words) >= 0.8:
                    found_coverage = True
                    break
        
        if not found_coverage:
            missing_sentences.append((i + 1, input_sentence))
    
    # Add analysis for missing sentences
    for sentence_num, missing_sentence in missing_sentences:
        logger.warning(f"Adding analysis for missing sentence {sentence_num}: {missing_sentence[:50]}...")
        missing_reason = "משפט זה לא נותח על ידי המודל הראשי" if language == "hebrew" else "This sentence was not analyzed by the main model"
        missing_analysis = _create_fallback_analysis_for_sentence(
            missing_sentence, 
            sentence_num, 
            missing_reason,
            language
        )
        validated_analysis.append(missing_analysis)
    
    # Step 5: Sort by sentence number to maintain order
    validated_analysis.sort(key=lambda x: x.get("sentence_number", 999))
    
    logger.info(f"Validation complete: {len(validated_analysis)} total analyses, {len(missing_sentences)} added for missing sentences")
    return validated_analysis

def generate_fact_check_with_search_grounding(
    user_event_id: int, 
    user_event_quote: str, 
    rss_quotes: str, 
    wiki_quotes: str, 
    correlation_json: str,
    # Additional comprehensive data from final report
    query: str = "",
    author: str = "",
    author_info: str = "",
    user_events: str = "",
    rss_payload: str = "",
    matches_dict: str = "",
    character_info: str = "",
    calibration_data: str = ""
) -> str:
    """
    Enhanced fact-check analysis with Hebrew language detection and Hebrew-first processing.
    This function follows the clean pipeline: Generate -> Validate -> Format -> Return
    """
    
    # Configure the API
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        logger.error("GEMINI_API_KEY environment variable not set")
        # Create fallback analysis
        input_sentences = split_sentences(user_event_quote)
        primary_language, _ = detect_language(user_event_quote)
        fallback_reason = "מפתח API חסר" if primary_language == "hebrew" else "API key missing"
        fallback_analysis = _create_complete_fallback_analysis(input_sentences, fallback_reason, primary_language)
        return json.dumps(fallback_analysis, ensure_ascii=False)
    
    # Create client with the new Google Gen AI SDK
    client = genai.Client(api_key=api_key)
    
    # STEP 1: Detect language and prepare input data
    input_sentences = split_sentences(user_event_quote)
    primary_language, confidence = detect_language(user_event_quote)
    
    logger.info(f"🎯 Starting fact-check for {len(input_sentences)} sentences")
    logger.info(f"🌐 Detected language: {primary_language} (confidence: {confidence:.2f})")
    
    # Automatically find relevant RSS articles using TF-IDF
    logger.info("Finding relevant RSS articles using TF-IDF similarity...")
    try:
        relevant_rss_articles = find_relevant_rss_articles(user_event_quote, max_articles=5)
        logger.info(f"Found relevant RSS articles: {len(relevant_rss_articles)} characters")
    except Exception as e:
        logger.warning(f"Error finding relevant RSS articles: {e}")
        relevant_rss_articles = "לא ניתן היה למצוא מקורות RSS רלוונטיים." if primary_language == "hebrew" else "Could not find relevant RSS sources."
    
    # Create sentence breakdown for the prompt
    sentence_breakdown = ""
    for i, sentence in enumerate(input_sentences, 1):
        sentence_breakdown += f"משפט {i}: {sentence}\n" if primary_language == "hebrew" else f"Sentence {i}: {sentence}\n"
    
    # Build comprehensive context
    context_parts = []
    
    if query:
        context_parts.append(f"Original Query: {query}")
    
    if author:
        context_parts.append(f"Author: {author}")
    
    if author_info:
        context_parts.append(f"Author Information: {author_info}")
    
    if user_events:
        context_parts.append(f"User Events: {user_events}")
    
    if rss_payload:
        context_parts.append(f"RSS Articles: {rss_payload}")
    
    if matches_dict:
        context_parts.append(f"Event Matches: {matches_dict}")
    
    if character_info:
        context_parts.append(f"Character Information: {character_info}")
    
    if calibration_data:
        context_parts.append(f"Calibration Data: {calibration_data}")
    
    comprehensive_context = "\n\n".join(context_parts)
    
    # Enhanced system prompt with Hebrew language support
    if primary_language == "hebrew":
        system_prompt = f"""
אתה בודק עובדות מקצועי המנתח טקסט בעברית.

🚨 דרישות קריטיות לפלט 🚨
- החזר רק מבנה JSON תקין - התחל עם [ וסיים עם ]
- נתח את כל {len(input_sentences)} המשפטים שסופקו
- כל משפט מקבל ציונים מרובים ואימות מקורות מפורט
- כל התגובות חייבות להיות בעברית!

🔍 דרישות חיפוש חובה 🔍
- אתה חייב להשתמש בכלי החיפוש Google Search
- אתה חייב לבדוק את נתוני RSS הקיימים
- אתה חייב לבדוק את נתוני ויקיפדיה הקיימים
- אסור לך לומר "אני צריך חיפוש נוסף" - השתמש בכלים הזמינים!
- בצע לפחות 2-3 שאילתות חיפוש שונות לכל משפט

תאריך ושעה נוכחיים: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} (שעון ישראל)

משפטים לניתוח ({len(input_sentences)} סה"כ):
{sentence_breakdown}

מערכת ציונים (כל הציונים 0.0-1.0):

1. **ציון אובייקטיביות** - עד כמה עובדתי לעומת דעה:
   - 1.0: עובדות טהורות (מספרים, תאריכים, אירועים)
   - 0.5: תערובת עובדתית ודעה
   - 0.0: דעה/סובייקטיבי טהור

2. **ציון התרחשות אירוע** - האם האירועים המתוארים התרחשו:
   - 1.0: עדויות חזקות ממקורות מרובים ומהימנים
   - 0.5: עדויות חלקיות או אישור עקיף
   - 0.0: אין עדויות או סותר עדויות

3. **ציון נכונות ציטוט** - דיוק הצהרות מצוטטות:
   - 1.0: ציטוט מדויק נמצא
   - 0.5: משמעות דומה נמצאה
   - 0.0: שונה/מומצא
   - null: אין ציטוטים במשפט

4. **ציון דמיון** - עד כמה המשפט תואם למקורות:
   - 1.0: התאמה מושלמת עם מקורות
   - 0.5: דמיון בינוני
   - 0.0: אין חיבור למקורות

5. **ציון דיוק כולל** - הערכת אימות סופית:
   - שקול את כל העדויות שנמצאו
   - הביא בחשבון איכות המקורות ועקביות
   - החל קנסות על פרשנות נסתרת/הטיה
   - **חובה: הפחת 0.2 נקודות או יותר אם תוכן שלילי מופיע בטקסט אך לא נמצא במקורות** - זה סימן לדיסאינפורמציה

פורמט JSON חובה:
החזר מערך עם בדיוק {len(input_sentences)} אובייקטים המכילים:

{{
  "sentence_text": "טקסט עברי מקורי",
  "sentence_number": 1,
  "classification": "not_s",
  "sentence_nature": "טענה ישירה מהמשתמש",
  "main_event_identified": "תיאור הטענה העיקרית",
  "arguments_extracted": ["טענה1", "טענה2"],
  "objectivity_score": 0.8,
  "event_happened_score": 0.7,
  "quote_correctness_score": 0.6,
  "overall_accuracy_score": 0.7,
  "similarity_score": 0.8,
  "hidden_interpretation_detected": false,
  "interpretation_details": null,
  "reasoning": "בדקתי את [שאילתה1, שאילתה2, שאילתה3], מצאתי בויקיפדיה/RSS/חיפוש [פרטים], יש התאמה ב[אם יש], יש פערים ב[אם יש]. [הסבר מפורט נוסף]",
  "sources": [
    {{
      "type": "Search",
      "source_name": "BBC עברית",
      "title": "כותרת המאמר",
      "url": "כתובת URL מלאה",
      "snippet": "טקסט רלוונטי",
      "date_published": "2024-01-01",
      "relevance_score": 0.9
    }}
  ],
  "search_queries_used": ["שאילתה1", "שאילתה2", "שאילתה3"],
  "low_score_explanation": "הסבר עברי אם ציון < 0.7"
}}

תהליך אימות חובה:
1. חלץ את כל הטענות העובדתיות מכל משפט
2. צור לפחות 2-3 שאילתות חיפוש ספציפיות בעברית לאימות
3. השתמש בכלי Google Search לחיפוש מידע עדכני - זה חובה!
4. בדוק ובדוק שוב את נתוני RSS וויקיפדיה שסופקו
5. דרג בהתבסס על איכות הראיות ועקביות
6. **זהה תוכן שלילי במשפט (הטבות, פגיעות, ביקורות) - אם לא נמצא תמיכה במקורות, הפחת מהציון הכולל 0.2 נקודות לפחות**
7. ספק נימוק מפורט בעברית בפורמט: "בדקתי את [כל השאילתות], מצאתי ב[מקורות], יש התאמה ב[אם יש], יש פערים ב[אם יש]"

זכור: החזר רק את מערך ה-JSON, ללא טקסט אחר. השתמש בכלים הזמינים!
"""
    else:  # English prompt
        system_prompt = f"""
You are a professional fact-checker analyzing text.

🚨 CRITICAL OUTPUT REQUIREMENT 🚨
- Return ONLY valid JSON format - start with [ and end with ]
- Analyze ALL {len(input_sentences)} sentences provided
- Each sentence gets multiple scores and detailed source verification

🔍 MANDATORY SEARCH REQUIREMENTS 🔍
- You MUST use the Google Search tool available to you
- You MUST check the provided RSS data thoroughly
- You MUST check the provided Wikipedia data thoroughly
- You are NOT allowed to say "I need more search" - use the available tools!
- Perform at least 2-3 different search queries per sentence

CURRENT DATE AND TIME: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} (Israel time)

INPUT SENTENCES TO ANALYZE ({len(input_sentences)} total):
{sentence_breakdown}

SCORING SYSTEM (all scores 0.0-1.0):

1. **OBJECTIVITY SCORE** - How factual vs opinion-based:
   - 1.0: Pure facts (numbers, dates, events)
   - 0.5: Mixed factual and opinion
   - 0.0: Pure opinion/subjective

2. **EVENT_HAPPENED SCORE** - Did described events occur:
   - 1.0: Strong multi-source evidence
   - 0.5: Partial/indirect evidence  
   - 0.0: No evidence or contradicted

3. **QUOTE_CORRECTNESS SCORE** - Accuracy of quoted statements:
   - 1.0: Exact quote found
   - 0.5: Similar meaning found
   - 0.0: Different/fabricated
   - null: No quotes in sentence

4. **SIMILARITY_SCORE** - How well sentence matches sources:
   - 1.0: Perfect match with sources
   - 0.5: Moderate similarity
   - 0.0: No connection to sources

5. **OVERALL_ACCURACY SCORE** - Final verification assessment:
   - Consider all evidence found
   - Factor in source quality and consistency
   - Apply penalties for hidden interpretation/bias
   - **MANDATORY: Reduce by 0.2 points or more if negative content appears in input text but not found in sources** - this is often a sign of misinformation

MANDATORY JSON OUTPUT FORMAT:
Return array with exactly {len(input_sentences)} objects containing:

{{
  "sentence_text": "original text",
  "sentence_number": 1,
  "classification": "not_s",
  "sentence_nature": "Direct Assertion by user_event_quote",
  "main_event_identified": "main claim description",
  "arguments_extracted": ["claim1", "claim2"],
  "objectivity_score": 0.8,
  "event_happened_score": 0.7,
  "quote_correctness_score": 0.6,
  "overall_accuracy_score": 0.7,
  "similarity_score": 0.8,
  "hidden_interpretation_detected": false,
  "interpretation_details": null,
  "reasoning": "I checked [query1, query2, query3], found in Wikipedia/RSS/Search [details], there are matches in [if any], there are gaps in [if any]. [Additional detailed explanation]",
  "sources": [
    {{
      "type": "Search",
      "source_name": "BBC News",
      "title": "Article Title",
      "url": "full URL",
      "snippet": "relevant text",
      "date_published": "2024-01-01",
      "relevance_score": 0.9
    }}
  ],
  "search_queries_used": ["query1", "query2", "query3"],
  "low_score_explanation": "explanation if score < 0.7"
}}

MANDATORY VERIFICATION PROCESS:
1. Extract all factual claims from each sentence
2. Generate at least 2-3 specific search queries for verification
3. Use Google Search tool to find current information - this is mandatory!
4. Cross-reference with provided RSS and Wikipedia data thoroughly
5. Score based on evidence quality and consistency
6. **Identify negative content in sentence (harms, injuries, criticisms) - if not supported by sources, reduce overall score by 0.2 points or less**
7. Provide detailed reasoning in format: "I checked [all queries], found in [sources], there are matches in [if any], there are gaps in [if any]"

Remember: Return ONLY the JSON array, no other text. Use the available tools!
"""
    
    # Create the main prompt with language-appropriate text
    if primary_language == "hebrew":
        main_prompt = f"""
{system_prompt}

אירוע משתמש #{user_event_id}: {user_event_quote}

ציטוטי RSS: {rss_quotes}

ציטוטי ויקיפדיה: {wiki_quotes}

נתוני קורלציה: {correlation_json}

הקשר נוסף:
{comprehensive_context}

מקורות RSS רלוונטיים:
{relevant_rss_articles}

בצע בדיקת עובדות מקיפה והחזר JSON בלבד בעברית.
"""
    else:
        main_prompt = f"""
{system_prompt}

User Event #{user_event_id}: {user_event_quote}

RSS Quotes: {rss_quotes}

Wikipedia Quotes: {wiki_quotes}

Correlation Data: {correlation_json}

Additional Context:
{comprehensive_context}

Relevant RSS Sources:
{relevant_rss_articles}

Perform comprehensive fact-checking and return JSON only.
"""
    
    try:
        # STEP 2: Generate LLM response (let it complete fully)
        logger.info("🤖 Generating LLM fact-check response...")
        
        def make_api_call():
            return client.models.generate_content(
                model='gemini-2.0-flash',
                contents=main_prompt,
                config=GenerateContentConfig(
                    temperature=0.1,  # Lower temperature for consistency
                    top_p=0.8,
                    top_k=40,
                    max_output_tokens=30000,
                    system_instruction=system_prompt,
                    safety_settings=[
                        SafetySetting(
                            category=HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                            threshold=HarmBlockThreshold.BLOCK_NONE
                        ),
                        SafetySetting(
                            category=HarmCategory.HARM_CATEGORY_HARASSMENT,
                            threshold=HarmBlockThreshold.BLOCK_NONE
                        ),
                        SafetySetting(
                            category=HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                            threshold=HarmBlockThreshold.BLOCK_NONE
                        ),
                        SafetySetting(
                            category=HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                            threshold=HarmBlockThreshold.BLOCK_NONE
                        ),
                    ],
                    tools=[
                        Tool(google_search=GoogleSearch())
                    ]
                )
            )
        
        # Generate response with retry logic
        response = retry_api_call(make_api_call, max_retries=3, base_delay=2.0)
        
        if not response or not response.text:
            logger.error("Empty or no response from LLM")
            error_msg = "לא התקבלה תגובה מהמודל" if primary_language == "hebrew" else "No response received from model"
            fallback_analysis = _create_complete_fallback_analysis(input_sentences, error_msg, primary_language)
            return json.dumps(fallback_analysis, ensure_ascii=False)
        
        raw_response = response.text.strip()
        logger.info(f"✅ LLM response generated: {len(raw_response)} characters")
        
        # STEP 3: Parse and validate the response (AFTER LLM completes)
        logger.info("🔍 Parsing and validating LLM response...")
        validated_analysis = _parse_and_validate_llm_response(raw_response, input_sentences, primary_language)
        
        # STEP 4: Format and return clean JSON
        final_json = json.dumps(validated_analysis, ensure_ascii=False, indent=2)
        logger.info(f"✅ Final validation complete: {len(validated_analysis)} sentence analyses ready in {primary_language}")
        
        return final_json
        
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Error in fact-check generation: {error_msg}")
        
        # Create comprehensive error fallback with language-appropriate text
        if primary_language == "hebrew":
            if '503' in error_msg or 'overloaded' in error_msg.lower():
                error_reason = "השירות עמוס כרגע"
            elif 'quota' in error_msg.lower() or 'rate limit' in error_msg.lower():
                error_reason = "הגעת למגבלת השימוש"
            else:
                error_reason = "שגיאה בבדיקת עובדות"
        else:
            if '503' in error_msg or 'overloaded' in error_msg.lower():
                error_reason = "Service currently overloaded"
            elif 'quota' in error_msg.lower() or 'rate limit' in error_msg.lower():
                error_reason = "Usage limit reached"
            else:
                error_reason = "Error in fact-checking"
        
        fallback_analysis = _create_complete_fallback_analysis(input_sentences, error_reason, primary_language)
        return json.dumps(fallback_analysis, ensure_ascii=False)

# Legacy functions for compatibility
def generate_fact_check(
    user_event_id: int, 
    user_event_quote: str, 
    rss_quotes: str, 
    wiki_quotes: str, 
    correlation_json: str,
    query: str = "",
    author: str = "",
    author_info: str = "",
    user_events: str = "",
    rss_payload: str = "",
    matches_dict: str = "",
    character_info: str = "",
    calibration_data: str = ""
) -> str:
    """
    Legacy function - now uses the enhanced search grounding version with Hebrew support.
    """
    return generate_fact_check_with_search_grounding(
        user_event_id, user_event_quote, rss_quotes, wiki_quotes, correlation_json,
        query, author, author_info, user_events, rss_payload, matches_dict, 
        character_info, calibration_data
    )

async def generate_fact_check_with_search(
    user_event_id: int, 
    user_event_quote: str, 
    rss_quotes: str, 
    wiki_quotes: str, 
    correlation_json: str,
    query: str = "",
    author: str = "",
    author_info: str = "",
    user_events: str = "",
    rss_payload: str = "",
    matches_dict: str = "",
    character_info: str = "",
    calibration_data: str = ""
) -> str:
    """
    Async version - uses the enhanced search grounding version with Hebrew support.
    """
    logger.info(f"Starting enhanced fact check with Hebrew language support for event {user_event_id}")
    logger.info(f"Author information: {author}")
    
    return generate_fact_check_with_search_grounding(
        user_event_id, user_event_quote, rss_quotes, wiki_quotes, correlation_json,
        query, author, author_info, user_events, rss_payload, matches_dict, 
        character_info, calibration_data
    )