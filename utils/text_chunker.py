#!/usr/bin/env python3
"""
Text chunking utility for handling long texts by splitting them into manageable chunks.
Splits text into approximately 250-word chunks, preferring paragraph boundaries.
"""

import re
from typing import List, Tuple


def count_words(text: str) -> int:
    """Count words in text, handling Hebrew and English."""
    # Remove extra whitespace and split by spaces
    words = text.strip().split()
    return len([word for word in words if word.strip()])


def split_into_sentences(text: str) -> List[str]:
    """Split text into sentences, handling Hebrew and English punctuation."""
    # Hebrew and English sentence endings
    sentence_endings = r'[.!?։…׃]+'
    
    # Split by sentence endings but keep the punctuation
    sentences = re.split(f'({sentence_endings})', text)
    
    # Recombine sentences with their punctuation
    result = []
    for i in range(0, len(sentences) - 1, 2):
        if i + 1 < len(sentences):
            sentence = sentences[i] + sentences[i + 1]
        else:
            sentence = sentences[i]
        
        sentence = sentence.strip()
        if sentence:
            result.append(sentence)
    
    return result


def split_into_paragraphs(text: str) -> List[str]:
    """Split text into paragraphs by double newlines."""
    paragraphs = re.split(r'\n\s*\n', text)
    return [p.strip() for p in paragraphs if p.strip()]


def chunk_text_by_words(text: str, target_words: int = 250, max_chunk_words: int = 300) -> List[Tuple[str, int]]:
    """
    Split text into chunks of approximately target_words, preferring paragraph boundaries.
    
    Args:
        text: The input text to chunk
        target_words: Target number of words per chunk (default: 250)
        max_chunk_words: Maximum words per chunk before forcing a split (default: 300)
    
    Returns:
        List of tuples: (chunk_text, word_count)
    """
    if not text or not text.strip():
        return []
    
    total_words = count_words(text)
    
    # If text is short enough, return as single chunk
    if total_words <= target_words:
        return [(text.strip(), total_words)]
    
    chunks = []
    
    # First, try to split by paragraphs
    paragraphs = split_into_paragraphs(text)
    
    if len(paragraphs) <= 1:
        # No clear paragraphs, try splitting by sentences
        sentences = split_into_sentences(text)
        return _chunk_by_sentences(sentences, target_words, max_chunk_words)
    
    # Process paragraphs
    current_chunk: List[str] = []
    current_word_count = 0
    
    for paragraph in paragraphs:
        paragraph_words = count_words(paragraph)
        
        # If adding this paragraph would exceed max_chunk_words, finalize current chunk
        if current_chunk and current_word_count + paragraph_words > max_chunk_words:
            chunk_text = '\n\n'.join(current_chunk).strip()
            chunks.append((chunk_text, current_word_count))
            current_chunk = []
            current_word_count = 0
        
        # If this paragraph alone exceeds target_words, split it further
        if paragraph_words > target_words:
            # Finalize current chunk if it exists
            if current_chunk:
                chunk_text = '\n\n'.join(current_chunk).strip()
                chunks.append((chunk_text, current_word_count))
                current_chunk = []
                current_word_count = 0
            
            # Split the large paragraph by sentences
            sentences = split_into_sentences(paragraph)
            sentence_chunks = _chunk_by_sentences(sentences, target_words, max_chunk_words)
            chunks.extend(sentence_chunks)
        else:
            # Add paragraph to current chunk
            current_chunk.append(paragraph)
            current_word_count += paragraph_words
            
            # If we've reached a good chunk size, finalize it
            if current_word_count >= target_words:
                chunk_text = '\n\n'.join(current_chunk).strip()
                chunks.append((chunk_text, current_word_count))
                current_chunk = []
                current_word_count = 0
    
    # Add remaining content as final chunk
    if current_chunk:
        chunk_text = '\n\n'.join(current_chunk).strip()
        chunks.append((chunk_text, current_word_count))
    
    return chunks


def _chunk_by_sentences(sentences: List[str], target_words: int, max_chunk_words: int) -> List[Tuple[str, int]]:
    """Helper function to chunk a list of sentences."""
    chunks = []
    current_chunk: List[str] = []
    current_word_count = 0
    
    for sentence in sentences:
        sentence_words = count_words(sentence)
        
        # If adding this sentence would exceed max_chunk_words, finalize current chunk
        if current_chunk and current_word_count + sentence_words > max_chunk_words:
            chunk_text = ' '.join(current_chunk).strip()
            chunks.append((chunk_text, current_word_count))
            current_chunk = []
            current_word_count = 0
        
        # Add sentence to current chunk
        current_chunk.append(sentence)
        current_word_count += sentence_words
        
        # If we've reached a good chunk size, finalize it
        if current_word_count >= target_words:
            chunk_text = ' '.join(current_chunk).strip()
            chunks.append((chunk_text, current_word_count))
            current_chunk = []
            current_word_count = 0
    
    # Add remaining sentences as final chunk
    if current_chunk:
        chunk_text = ' '.join(current_chunk).strip()
        chunks.append((chunk_text, current_word_count))
    
    return chunks


def prepare_chunk_with_author(chunk_text: str, author_name: str = "", chunk_index: int = 1, total_chunks: int = 1) -> str:
    """
    Prepare a text chunk with author information and chunk metadata.
    
    Args:
        chunk_text: The text chunk
        author_name: Author name to include (if provided)
        chunk_index: Current chunk number (1-based)
        total_chunks: Total number of chunks
    
    Returns:
        Formatted text ready for processing
    """
    result = ""
    
    if author_name:
        result += f"Author: {author_name}\n\n"
    
    if total_chunks > 1:
        result += f"[Part {chunk_index} of {total_chunks}]\n\n"
    
    result += chunk_text
    
    return result


# Test function
if __name__ == "__main__":
    # Test with Hebrew text
    hebrew_test = """
היועצת המשפטית לממשלה פוגעת בביטחון המדינה ומונעת מינויים חשובים. זה נושא רציני מאוד שצריך לטפל בו.

מסיתים כמוכם הביאו לרצח רבין, לא למדתם כלום. יש לכם אפס הערכה לאנשים שהקריבו את חייהם עבור המדינה הזאת.

כשאתם ישבתם בממדים בשבעה באוקטובר אני יצאתי לנובה להציל אנשים. יצאתי לסכן את חיי, ואתם ישבתם בבית לבטח ועכשיו צועקים לי בוגד.

אין בכם כלום חוץ משנאה, אתם לא יודעים שום דבר חוץ משנאה. זה מה שיצא מכנס שדרות שכותרתו איך נבנים מחדש כשעדיין לא הפסקנו להתפרק. זה מה שאתם עושים - פירוק מדינת ישראל.
    """.strip()
    
    print("Testing text chunking:")
    print(f"Original text ({count_words(hebrew_test)} words):")
    print(hebrew_test[:100] + "...")
    print()
    
    chunks = chunk_text_by_words(hebrew_test, target_words=50)  # Small chunks for testing
    
    for i, (chunk, word_count) in enumerate(chunks, 1):
        print(f"Chunk {i} ({word_count} words):")
        print(chunk[:100] + "..." if len(chunk) > 100 else chunk)
        print() 