#!/usr/bin/env python3
"""
Fact Checker Verifier - OpenAI GPT-based verification and completion system
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional

from openai import OpenAI

# Import existing fact checker functions
from .llm_fact_checker import (
    create_missing_sentence_analysis,
    generate_fact_check_with_search_grounding,
    split_sentences,
    validate_sentence_coverage,
)

logger = logging.getLogger(__name__)


class FactCheckerVerifier:
    """
    OpenAI GPT-based verifier that ensures complete fact-checking coverage.
    """
    
    def __init__(self):
        # Use the simplest possible OpenAI client initialization
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        if not os.environ.get("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY environment variable not set")
    
    def verify_and_complete_analysis(
        self,
        user_event_id: int,
        user_event_quote: str,
        fact_check_result: str,
        rss_quotes: str = "",
        wiki_quotes: str = "",
        correlation_json: str = "",
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
        Verify fact-checking completeness and fill in missing analyses.
        
        Args:
            user_event_id: Event ID
            user_event_quote: Original text to fact-check
            fact_check_result: JSON result from primary fact checker
            ... other context data ...
            
        Returns:
            Complete, verified JSON analysis
        """
        
        logger.info(f"Starting verification for event {user_event_id}")
        
        # Split original text into sentences
        input_sentences = split_sentences(user_event_quote)
        logger.info(f"Original text split into {len(input_sentences)} sentences")
        
        try:
            # Parse fact checker result
            parsed_result = json.loads(fact_check_result)
            logger.info(f"Fact checker returned {len(parsed_result)} analyses")
            
            # Validate coverage
            validation_result = validate_sentence_coverage(input_sentences, parsed_result)
            logger.info(f"Coverage validation: {validation_result['coverage_percentage']:.1f}%")
            
            if validation_result["all_covered"]:
                logger.info("All sentences covered, verifying completeness...")
                return self._verify_completeness(user_event_quote, parsed_result, input_sentences)
            else:
                logger.warning(f"Missing {len(validation_result['missing_sentences'])} sentences")
                return self._handle_missing_sentences(
                    user_event_id, user_event_quote, parsed_result, validation_result,
                    rss_quotes, wiki_quotes, correlation_json, query, author, author_info,
                    user_events, rss_payload, matches_dict, character_info, calibration_data
                )
                
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON from fact checker: {e}")
            logger.info("Recreating full analysis due to JSON error")
            return self._recreate_full_analysis(
                user_event_id, user_event_quote, rss_quotes, wiki_quotes, correlation_json,
                query, author, author_info, user_events, rss_payload, matches_dict,
                character_info, calibration_data
            )
    
    def _verify_completeness(self, user_event_quote: str, parsed_result: List[Dict], input_sentences: List[str]) -> str:
        """
        Verify that all analyses are complete and properly formatted.
        """
        
        # Check if Hebrew input
        has_hebrew = any('\u0590' <= char <= '\u05FF' for char in user_event_quote)
        
        # Required fields for each analysis
        required_fields = [
            'sentence_text', 'sentence_number', 'classification', 'sentence_nature',
            'main_event_identified', 'arguments_extracted', 'objectivity_score',
            'event_happened_score', 'quote_correctness_score', 'overall_accuracy_score',
            'similarity_score', 'hidden_interpretation_detected', 'interpretation_details',
            'reasoning', 'sources', 'search_queries_used', 'low_score_explanation'
        ]
        
        incomplete_analyses = []
        
        for i, analysis in enumerate(parsed_result):
            missing_fields = []
            for field in required_fields:
                if field not in analysis:
                    missing_fields.append(field)
                elif field == 'reasoning' and has_hebrew:
                    # Check if reasoning is in Hebrew when input is Hebrew
                    reasoning = analysis.get('reasoning', '')
                    if reasoning and not any('\u0590' <= char <= '\u05FF' for char in reasoning):
                        missing_fields.append(f'{field}_hebrew')
            
            if missing_fields:
                incomplete_analyses.append({
                    'index': i,
                    'sentence': analysis.get('sentence_text', ''),
                    'missing_fields': missing_fields
                })
        
        if incomplete_analyses:
            logger.info(f"Found {len(incomplete_analyses)} incomplete analyses, completing with OpenAI...")
            return self._complete_analyses_with_openai(user_event_quote, parsed_result, incomplete_analyses, has_hebrew)
        else:
            logger.info("All analyses are complete")
            return json.dumps(parsed_result, ensure_ascii=False)
    
    def _complete_analyses_with_openai(self, user_event_quote: str, parsed_result: List[Dict], incomplete_analyses: List[Dict], has_hebrew: bool) -> str:
        """
        Use OpenAI to complete missing fields in analyses.
        """
        
        system_prompt = """
🎯 Role: Fact-Checking Verification & Completion Agent

You are a systematic, precise verifier ensuring complete fact-checking analyses. 

📋 TASK:
Complete missing fields in fact-checking analyses based on provided context and scoring guidelines.

🌐 LANGUAGE REQUIREMENT:
""" + ("**CRITICAL: Input is in Hebrew - ALL responses must be in Hebrew**" if has_hebrew else "Respond in English") + """

⭐ SCORING GUIDELINES:
- objectivity_score: 1.0=pure facts, 0.0=pure opinion
- event_happened_score: 1.0=strong evidence, 0.0=no evidence  
- quote_correctness_score: 1.0=exact quote, 0.0=fabricated
- overall_accuracy_score: 1.0=well supported, 0.0=fake news
- similarity_score: 1.0=perfect match to sources, 0.0=no connection
- hidden_interpretation_detected: true if subjective characterizations without attribution
- reasoning: 15-25 words summary
- low_score_explanation: required if overall_accuracy_score < 0.8

Return the complete JSON array with all fields filled.
"""
        
        user_prompt = f"""
Original text: {user_event_quote}

Current analyses with missing fields:
{json.dumps(parsed_result, ensure_ascii=False, indent=2)}

Incomplete analyses details:
{json.dumps(incomplete_analyses, ensure_ascii=False, indent=2)}

Please complete all missing fields following the scoring guidelines and language requirements.
"""
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.3,
                max_tokens=8000
            )
            
            completed_result = response.choices[0].message.content
            
            # Validate the completed result
            try:
                completed_parsed = json.loads(completed_result)
                if 'analyses' in completed_parsed:
                    completed_parsed = completed_parsed['analyses']
                    
                logger.info("OpenAI successfully completed missing fields")
                return json.dumps(completed_parsed, ensure_ascii=False)
                
            except json.JSONDecodeError:
                logger.error("OpenAI returned invalid JSON, using original with fallbacks")
                return self._apply_fallback_completion(parsed_result, incomplete_analyses, has_hebrew)
                
        except Exception as e:
            logger.error(f"OpenAI completion failed: {e}")
            return self._apply_fallback_completion(parsed_result, incomplete_analyses, has_hebrew)
    
    def _apply_fallback_completion(self, parsed_result: List[Dict], incomplete_analyses: List[Dict], has_hebrew: bool) -> str:
        """
        Apply fallback completion for missing fields.
        """
        
        for incomplete in incomplete_analyses:
            index = incomplete['index']
            analysis = parsed_result[index]
            
            # Fill missing fields with reasonable defaults
            if 'objectivity_score' not in analysis:
                analysis['objectivity_score'] = 0.5
            if 'event_happened_score' not in analysis:
                analysis['event_happened_score'] = 0.5
            if 'quote_correctness_score' not in analysis:
                analysis['quote_correctness_score'] = None
            if 'overall_accuracy_score' not in analysis:
                analysis['overall_accuracy_score'] = 0.5
            if 'similarity_score' not in analysis:
                analysis['similarity_score'] = 0.5
            if 'hidden_interpretation_detected' not in analysis:
                analysis['hidden_interpretation_detected'] = False
            if 'interpretation_details' not in analysis:
                analysis['interpretation_details'] = None
            if 'reasoning' not in analysis or ('reasoning_hebrew' in incomplete['missing_fields']):
                if has_hebrew:
                    analysis['reasoning'] = "ניתוח בסיסי - שדות חסרים הושלמו אוטומטית"
                else:
                    analysis['reasoning'] = "Basic analysis - missing fields completed automatically"
            if 'sources' not in analysis:
                analysis['sources'] = []
            if 'search_queries_used' not in analysis:
                analysis['search_queries_used'] = []
            if 'low_score_explanation' not in analysis and analysis.get('overall_accuracy_score', 1.0) < 0.8:
                if has_hebrew:
                    analysis['low_score_explanation'] = "ניתוח לא מלא - חסרים נתונים ומקורות לאימות"
                else:
                    analysis['low_score_explanation'] = "Incomplete analysis - missing data and sources for verification"
        
        logger.info("Applied fallback completion for missing fields")
        return json.dumps(parsed_result, ensure_ascii=False)
    
    def _handle_missing_sentences(
        self, user_event_id: int, user_event_quote: str, parsed_result: List[Dict],
        validation_result: Dict, *args
    ) -> str:
        """
        Handle missing sentences by sending them back to fact checker.
        """
        
        missing_sentences = validation_result['missing_sentences']
        logger.info(f"Sending {len(missing_sentences)} missing sentences back to fact checker")
        
        # Create text with only missing sentences
        missing_text = '\n'.join([sent['text'] for sent in missing_sentences])
        
        try:
            # Send missing sentences to fact checker
            missing_analysis_result = generate_fact_check_with_search_grounding(
                user_event_id, missing_text, *args
            )
            
            # Parse and integrate missing analyses
            missing_analyses = json.loads(missing_analysis_result)
            
            # Combine with existing analyses
            all_analyses = parsed_result + missing_analyses
            
            # Sort by sentence number if available
            all_analyses.sort(key=lambda x: x.get('sentence_number', 0))
            
            logger.info(f"Successfully integrated {len(missing_analyses)} missing analyses")
            return json.dumps(all_analyses, ensure_ascii=False)
            
        except Exception as e:
            logger.error(f"Failed to analyze missing sentences: {e}")
            # Use fallback for missing sentences
            fallback_analyses = create_missing_sentence_analysis(missing_sentences)
            all_analyses = parsed_result + fallback_analyses
            return json.dumps(all_analyses, ensure_ascii=False)
    
    def _recreate_full_analysis(self, user_event_id: int, user_event_quote: str, *args) -> str:
        """
        Recreate full analysis when original result is completely invalid.
        """
        
        logger.info("Recreating full analysis due to invalid original result")
        
        try:
            new_result = generate_fact_check_with_search_grounding(
                user_event_id, user_event_quote, *args
            )
            
            # Verify the new result
            json.loads(new_result)  # Validate JSON
            logger.info("Successfully recreated full analysis")
            return new_result
            
        except Exception as e:
            logger.error(f"Failed to recreate analysis: {e}")
            # Create basic fallback for all sentences
            input_sentences = split_sentences(user_event_quote)
            fallback_analyses = create_missing_sentence_analysis([
                {"index": i+1, "text": sent} for i, sent in enumerate(input_sentences)
            ])
            return json.dumps(fallback_analyses, ensure_ascii=False)


# Global instance
verifier = FactCheckerVerifier()


def verify_fact_check_completeness(
    user_event_id: int,
    user_event_quote: str, 
    fact_check_result: str,
    rss_quotes: str = "",
    wiki_quotes: str = "",
    correlation_json: str = "",
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
    Main function to verify and complete fact-checking analysis.
    
    Returns:
        Complete, verified JSON analysis
    """
    
    return verifier.verify_and_complete_analysis(
        user_event_id, user_event_quote, fact_check_result,
        rss_quotes, wiki_quotes, correlation_json, query, author, author_info,
        user_events, rss_payload, matches_dict, character_info, calibration_data
    ) 