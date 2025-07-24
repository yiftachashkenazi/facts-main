#!/usr/bin/env python3
# File: wiki/wiki_knowledge.py
"""
Wikipedia knowledge processing module.
- Generates a summary of extracted entities.
- Computes similarity between user input and entity content using sentence transformers.
- Extracts three sentences before and after for entities with similarity > 0.8.
- Saves results to wiki_knowledge.sql.
- Returns structured data for the main pipeline.
"""

import json
import logging
import os
from pathlib import Path
from typing import List, Dict, Any
from dotenv import load_dotenv
import numpy as np
from sentence_transformers import SentenceTransformer
import re

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
MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"  # Matches rss_processor.py
SIMILARITY_THRESHOLD = 0.1  # Threshold for high similarity
KNOWLEDGE_SQL_FILENAME = "wiki_knowledge.sql"

# ——————————————————————————————————————————— #
# Helpers
# ——————————————————————————————————————————— #
def cosine(u: np.ndarray, v: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-8)

def extract_context_sentences(text: str, entity_name: str) -> Dict[str, str]:
    """
    Extract three sentences before and after the first occurrence of entity_name.
    Returns a dict with 'before' and 'after' sentences.
    """
    if not text or not entity_name:
        return {"before": "", "after": ""}

    # Split text into sentences
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]

    # Find the first sentence containing the entity_name
    target_idx = -1
    for idx, sentence in enumerate(sentences):
        if entity_name.lower() in sentence.lower():
            target_idx = idx
            break

    if target_idx == -1:
        logger.debug(f"Entity {entity_name} not found in text")
        return {"before": "", "after": ""}

    # Extract three sentences before and after
    start_idx = max(0, target_idx - 3)
    end_idx = min(len(sentences), target_idx + 4)  # +4 to include the target sentence
    before_sentences = sentences[start_idx:target_idx]
    after_sentences = sentences[target_idx + 1:end_idx]

    return {
        "before": " ".join(before_sentences),
        "after": " ".join(after_sentences)
    }

def generate_entity_summary(entities: List[Dict]) -> str:
    """Generate a summary of all entities."""
    summary_lines = ["Summary of Extracted Wikipedia Entities:"]
    for entity in entities:
        name = entity.get("name", "N/A")
        name_english = entity.get("name_english", "N/A")
        class_type = entity.get("class", "N/A")
        summary_text = entity.get("summary", "")[:200] + "..." if entity.get("summary") else "No summary available"
        summary_lines.append(
            f"- {name} ({name_english}, {class_type}): {summary_text}"
        )
    return "\n".join(summary_lines)

# ——————————————————————————————————————————— #
# Main Knowledge Processing Function
# ——————————————————————————————————————————— #
def process_wiki_knowledge(
    user_text: str,
    wiki_results: List[Dict],
    sql_output: str = KNOWLEDGE_SQL_FILENAME
) -> Dict[str, Any]:
    """
    Process Wikipedia entities to generate summary and find similar entities.
    - Summarizes all entities.
    - Computes similarity between user input and entity content.
    - Extracts context for entities with similarity > 0.8.
    - Saves results to SQL file.
    - Returns structured data for the pipeline.
    """
    if user_text is None:
        logger.error("process_wiki_knowledge called with None user_text")
        return {"summary": "", "similar_entities": []}

    # Detect Hebrew characters in user_text to indicate language
    if re.search(r'[\u0590-\u05FF]', user_text):
        logger.debug("Detected Hebrew input text; ensure wiki_results summaries are in Hebrew")
    else:
        logger.debug("Detected non-Hebrew input text")

    load_dotenv()
    logger.debug("Starting wiki knowledge processing")

    # Validate inputs
    if not user_text or not user_text.strip():
        logger.warning("Empty or invalid user_text provided")
        return {"summary": "", "similar_entities": []}
    if not wiki_results:
        logger.warning("No wiki results provided")
        return {"summary": "", "similar_entities": []}

    # Initialize model
    try:
        model = SentenceTransformer(MODEL_NAME)
        logger.debug(f"Model loaded: {MODEL_NAME}")
    except Exception as exc:
        logger.error(f"Failed to load model {MODEL_NAME}: {exc}")
        return {"summary": "", "similar_entities": []}

    # Encode user text
    try:
        user_vec = model.encode(user_text)
        logger.debug(f"User text encoded, embedding shape: {user_vec.shape}")
    except Exception as exc:
        logger.error(f"Failed to encode user_text: {exc}")
        return {"summary": "", "similar_entities": []}

    # Generate summary
    summary = generate_entity_summary(wiki_results)
    logger.debug("Generated entity summary")

    # Compute similarities and extract context
    similar_entities = []
    for entity in wiki_results:
        # Combine summary and full text for embedding
        content = (entity.get("summary", "") + " " + entity.get("full_text", "")).strip()
        if not content:
            logger.debug(f"No content for entity {entity.get('name', 'N/A')}")
            continue

        try:
            content_vec = model.encode(content)
            similarity = cosine(user_vec, content_vec)
            logger.debug(f"Similarity for {entity.get('name', 'N/A')}: {similarity}")
        except Exception as exc:
            logger.error(f"Failed to encode content for {entity.get('name', 'N/A')}: {exc}")
            continue

        if similarity > SIMILARITY_THRESHOLD:
            context = extract_context_sentences(entity.get("full_text", ""), entity.get("name_english", ""))
            similar_entities.append({
                "entity_id": entity.get("entity_id"),
                "name": entity.get("name"),
                "name_english": entity.get("name_english"),
                "class": entity.get("class"),
                "similarity": round(float(similarity), 4),
                "context_before": context["before"],
                "context_after": context["after"]
            })

    # Write to SQL
    sql_path = Path(sql_output).resolve()
    with open(sql_path, "w", encoding="utf-8") as f:
        # Summary table
        f.write("\nCREATE TABLE IF NOT EXISTS wiki_knowledge_summary (\n"
                "    summary_id VARCHAR(255) PRIMARY KEY,\n"
                "    summary_text TEXT\n"
                ");\n\n")
        summary_id = "summary_1"
        summary_sql = summary.replace("'", "''")
        f.write("INSERT INTO wiki_knowledge_summary (summary_id, summary_text) VALUES (\n"
                f"    '{summary_id}',\n"
                f"    '{summary_sql}'\n"
                ");\n\n")

        # Similar entities table
        f.write("\nCREATE TABLE IF NOT EXISTS wiki_knowledge_similar_entities (\n"
                "    entity_id VARCHAR(255) PRIMARY KEY,\n"
                "    name TEXT,\n"
                "    name_english TEXT,\n"
                "    class TEXT,\n"
                "    similarity FLOAT,\n"
                "    context_before TEXT,\n"
                "    context_after TEXT\n"
                ");\n\n")

        for entity in similar_entities:
            entity_id = entity["entity_id"].replace("'", "''")
            name = entity["name"].replace("'", "''")
            name_english = entity["name_english"].replace("'", "''")
            class_type = entity["class"].replace("'", "''")
            similarity = entity["similarity"]
            context_before = entity["context_before"].replace("'", "''")
            context_after = entity["context_after"].replace("'", "''")
            f.write("INSERT INTO wiki_knowledge_similar_entities (entity_id, name, name_english, class, similarity, context_before, context_after) VALUES (\n"
                    f"    '{entity_id}',\n"
                    f"    '{name}',\n"
                    f"    '{name_english}',\n"
                    f"    '{class_type}',\n"
                    f"    {similarity},\n"
                    f"    '{context_before}',\n"
                    f"    '{context_after}'\n"
                    ");\n\n")

    print(f"✔  Wiki knowledge SQL written to {sql_path}")
    print(f"Processed {len(similar_entities)} similar entities with similarity > {SIMILARITY_THRESHOLD}")

    return {
        "summary": summary,
        "similar_entities": similar_entities
    }

if __name__ == "__main__":
    # Simple test
    sample_text = (
        "ישראל שחררה את האסירים אתמול מעזה.\n"
        "לדעתי זה היה צעד בלתי-חוקי.\n"
        "הכנסת אישרה את חוק ההסדרה החדש ברוב גדול."
    )
    sample_wiki_results = [
        {
            "entity_id": "עזה",
            "name": "עזה",
            "name_english": "Gaza",
            "class": "entity",
            "summary": "Gaza is a coastal region in the Middle East.",
            "full_text": "Gaza is a coastal region. It has a complex history. Israel has controlled parts of Gaza since 1967. Recent events include conflicts and blockades. The region faces humanitarian challenges.",
            "url": "https://en.wikipedia.org/wiki/Gaza"
        }
    ]
    results = process_wiki_knowledge(
        user_text=sample_text,
        wiki_results=sample_wiki_results,
        sql_output="wiki_knowledge_test.sql"
    )
    print(f"Summary:\n{results['summary']}")
    print(f"Similar entities: {len(results['similar_entities'])}")