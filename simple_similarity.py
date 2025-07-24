#!/usr/bin/env python3

import logging
import math
import re
from collections import Counter
from typing import Dict, List

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger(__name__)

class SentenceTransformer:
    """
    Simple text similarity model using TF-IDF instead of sentence transformers.
    """
    
    def __init__(self, model_name: str = "simple"):
        logger.info(f"Creating SimpleSimilarityModel (ignoring model_name: {model_name})")
        self.vectorizer = TfidfVectorizer(
            max_features=10000,
            stop_words='english',
            ngram_range=(1, 2),
            analyzer='word'
        )
        self.corpus = []
        self.corpus_vectors = None
        logger.info("Initialized SimpleSimilarityModel")
    
    def add_to_corpus(self, texts: List[str]):
        """Add texts to the corpus for TF-IDF calculation."""
        if not texts:
            return
        
        # Clean and normalize texts
        cleaned_texts = []
        for text in texts:
            if text and isinstance(text, str):
                # Basic cleaning
                text = text.lower().strip()
                text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
                cleaned_texts.append(text)
        
        if cleaned_texts:
            self.corpus.extend(cleaned_texts)
            # Update TF-IDF vectors
            self.corpus_vectors = self.vectorizer.fit_transform(self.corpus)
    
    def encode(self, text: str) -> np.ndarray:
        """
        Encode text into a TF-IDF vector.
        """
        if not text or not isinstance(text, str):
            # Return zero vector if vectorizer is not fitted
            if not hasattr(self.vectorizer, 'vocabulary_') or self.vectorizer.vocabulary_ is None:
                return np.zeros(1000, dtype=np.float32)  # Default size
            return np.zeros(len(self.vectorizer.get_feature_names_out()), dtype=np.float32)
        
        # Clean and normalize text
        text = text.lower().strip()
        text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
        
        # If vectorizer is not fitted, fit it with the current text
        if not hasattr(self.vectorizer, 'vocabulary_') or self.vectorizer.vocabulary_ is None:
            self.vectorizer.fit([text])
        
        # Transform text to TF-IDF vector
        try:
            vector = self.vectorizer.transform([text])
            return vector.toarray()[0]
        except ValueError:
            # If text contains unknown words, refit with current corpus + new text
            if self.corpus:
                self.vectorizer.fit(self.corpus + [text])
            else:
                self.vectorizer.fit([text])
            vector = self.vectorizer.transform([text])
            return vector.toarray()[0]

def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    if len(vec1) != len(vec2):
        raise ValueError("Vectors must have the same length")
    
    # Calculate dot product
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    
    # Calculate magnitudes
    magnitude1 = math.sqrt(sum(a * a for a in vec1))
    magnitude2 = math.sqrt(sum(a * a for a in vec2))
    
    # Avoid division by zero
    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0
    
    return dot_product / (magnitude1 * magnitude2) 