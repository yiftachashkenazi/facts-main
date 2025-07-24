import wikipedia
import re
from collections import Counter

# הגדר שפת ברירת מחדל
wikipedia.set_lang("he")

def extract_candidate_phrases(text: str, named_entities: list, ngrams: dict) -> dict:
    # סינון ישויות שמיות באורך 2 מילים לפחות
    entity_phrases = [ent for ent in named_entities if len(ent.split()) >= 2]

    # הוצאת n-grams של 4 ו־3 מילים (בסדר יורד)
    fourgrams = ngrams.get("4-gram", [])
    trigrams = ngrams.get("3-gram", [])
    candidate_ngrams = fourgrams + trigrams

    # ניקוי כפילויות + העדפה לפי סדר הופעה
    seen = set()
    cleaned = []
    for phrase in entity_phrases + candidate_ngrams:
        if phrase not in seen:
            seen.add(phrase)
            cleaned.append(phrase)

    return {
        "entities": entity_phrases[:5],
        "ngrams": [ng for ng in candidate_ngrams if ng not in entity_phrases][:10]
    }

def search_wikipedia_for_phrases(phrases: list) -> dict:
    found = {}
    for phrase in phrases:
        try:
            results = wikipedia.search(phrase, results=1)
            if results:
                page = wikipedia.page(results[0])
                found[phrase] = {
                    "title": page.title,
                    "url": page.url,
                    "content": page.content,
                    "summary": page.summary
                }
        except Exception:
            continue
    return found

def select_top_phrases(entity_phrases, ngram_phrases, wiki_results) -> tuple:
    selected_entities = []
    selected_ngrams = []

    # בחירת ישויות שמופיעות בויקיפדיה
    for ent in entity_phrases:
        if ent in wiki_results:
            selected_entities.append(ent)
        if len(selected_entities) == 10:
            break

    # סינון ngrams שמופיעים בויקיפדיה
    # Filter only ngrams that exist as entries in Wikipedia
    ngram_phrases = [ng for ng in ngram_phrases if ng in wiki_results]

    # בחירת ngrams שמופיעים וגם מופיעים יחד עם אחרים וגם נמצאים כתוצאה בויקיפדיה
    remaining = []
    for ng in ngram_phrases:
        if ng in wiki_results:
            co_occur = sum(1 for other in ngram_phrases if other != ng and other in wiki_results and other in wiki_results[ng]['content'])
            remaining.append((ng, co_occur))
    
    # מיון לפי הופעה משותפת ואז לפי אורך
    remaining = sorted(remaining, key=lambda x: (-x[1], -len(x[0].split())))
    selected_ngrams = [r[0] for r in remaining[:10]]

    return selected_entities, selected_ngrams