import html
import json
import logging
import os
import re
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

class RSSArticle:
    """Represents an RSS article with metadata."""
    
    def __init__(self, title: str, link: str, published: str, summary: str, 
                 content: str, source: str, collected_at: str):
        self.title = title
        self.link = link
        self.published = published
        self.summary = summary
        self.content = content
        self.source = source
        self.collected_at = collected_at
        
        # Clean and extract text content
        self.clean_text = self._clean_html_content()
        
    def _clean_html_content(self) -> str:
        """Clean HTML content and extract readable text."""
        # Combine title, summary, and content
        full_text = f"{self.title} {self.summary} {self.content}"
        
        # Remove HTML tags
        clean_text = re.sub(r'<[^>]+>', ' ', full_text)
        
        # Decode HTML entities
        clean_text = html.unescape(clean_text)
        
        # Remove extra whitespace and normalize
        clean_text = re.sub(r'\s+', ' ', clean_text).strip()
        
        return clean_text
    
    def get_publication_year(self) -> Optional[int]:
        """Extract publication year from the published date."""
        try:
            # Check if published date is None or empty
            if not self.published or self.published == 'None' or not self.published.strip():
                return None
                
            # Try different date formats
            date_formats = [
                "%a, %d %b %Y %H:%M:%S %z",  # RFC 2822 format
                "%Y-%m-%dT%H:%M:%S%z",       # ISO format
                "%Y-%m-%d %H:%M:%S",         # Simple format
            ]
            
            for fmt in date_formats:
                try:
                    parsed_date = datetime.strptime(self.published, fmt)
                    return parsed_date.year
                except ValueError:
                    continue
                    
            # If no format works, try to extract year with regex
            year_match = re.search(r'20\d{2}', self.published)
            if year_match:
                return int(year_match.group())
                
        except Exception as e:
            logger.debug(f"Could not parse date '{self.published}': {e}")
            
        return None
    
    def to_dict(self) -> Dict:
        """Convert article to dictionary for JSON serialization."""
        return {
            "title": self.title,
            "link": self.link,
            "published": self.published,
            "summary": self.summary[:500] + "..." if len(self.summary) > 500 else self.summary,
            "source": self.source,
            "clean_text": self.clean_text[:1000] + "..." if len(self.clean_text) > 1000 else self.clean_text
        }

class RSSTFIDFMatcher:
    """TF-IDF based RSS article matcher for fact-checking."""
    
    def __init__(self, rss_articles_path: str = "data/rss_articles.json"):
        self.rss_articles_path = rss_articles_path
        self.articles: List[RSSArticle] = []
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.tfidf_matrix = None
        
        # Load articles on initialization
        self.load_articles()
        self.build_tfidf_index()
    
    def load_articles(self) -> None:
        """Load RSS articles from JSON file."""
        try:
            if not os.path.exists(self.rss_articles_path):
                logger.warning(f"RSS articles file not found: {self.rss_articles_path}")
                return
                
            with open(self.rss_articles_path, 'r', encoding='utf-8') as f:
                articles_data = json.load(f)
            
            logger.info(f"Loading {len(articles_data)} articles from RSS file...")
            
            # Filter articles from 2024 onwards
            filtered_articles = []
            for article_data in articles_data:
                article = RSSArticle(
                    title=article_data.get('title', ''),
                    link=article_data.get('link', ''),
                    published=article_data.get('published', ''),
                    summary=article_data.get('summary', ''),
                    content=article_data.get('content', ''),
                    source=article_data.get('source', ''),
                    collected_at=article_data.get('collected_at', '')
                )
                
                # Only include articles from 2024 onwards
                pub_year = article.get_publication_year()
                if pub_year and pub_year >= 2024:
                    filtered_articles.append(article)
            
            self.articles = filtered_articles
            logger.info(f"Loaded {len(self.articles)} articles from 2024 onwards")
            
        except Exception as e:
            logger.error(f"Error loading RSS articles: {e}")
            self.articles = []
    
    def build_tfidf_index(self) -> None:
        """Build TF-IDF index for all articles."""
        if not self.articles:
            logger.warning("No articles available to build TF-IDF index")
            return
            
        try:
            # Extract clean text from all articles
            documents = [article.clean_text for article in self.articles]
            
            # Create TF-IDF vectorizer with support for Hebrew and English
            self.vectorizer = TfidfVectorizer(
                max_features=10000,
                stop_words=None,  # Don't use English stop words for multilingual support
                ngram_range=(1, 2),  # Use unigrams and bigrams
                min_df=2,  # Ignore terms that appear in less than 2 documents
                max_df=0.8,  # Ignore terms that appear in more than 80% of documents
                lowercase=True,
                token_pattern=r'(?u)\b\w+\b'  # Support Unicode characters for Hebrew
            )
            
            # Fit and transform documents
            self.tfidf_matrix = self.vectorizer.fit_transform(documents)
            
            if self.tfidf_matrix is not None:
                logger.info(f"Built TF-IDF index with {self.tfidf_matrix.shape[0]} documents and {self.tfidf_matrix.shape[1]} features")
            
        except Exception as e:
            logger.error(f"Error building TF-IDF index: {e}")
            self.vectorizer = None
            self.tfidf_matrix = None
    
    def find_relevant_articles(self, query_text: str, top_k: int = 5) -> List[Tuple[RSSArticle, float]]:
        """Find the most relevant articles for a given query text using TF-IDF similarity."""
        if not self.vectorizer or self.tfidf_matrix is None:
            logger.warning("TF-IDF index not available")
            return []
        
        try:
            logger.info(f"🔍 TF-IDF Search: Looking for relevant articles...")
            logger.info(f"   📝 Query: {query_text[:100]}...")
            logger.info(f"   📊 Available articles: {len(self.articles)}")
            logger.info(f"   🎯 Requesting top {top_k} results")
            
            # Transform query text using the same vectorizer
            query_vector = self.vectorizer.transform([query_text])
            
            # Calculate cosine similarity between query and all articles
            similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
            
            # Get top-k most similar articles
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            # Return articles with their similarity scores
            results = []
            for idx in top_indices:
                if similarities[idx] > 0:  # Only include articles with positive similarity
                    results.append((self.articles[idx], float(similarities[idx])))
            
            # Log detailed results
            logger.info(f"✅ TF-IDF Results:")
            logger.info(f"   📈 Found {len(results)} relevant articles")
            if results:
                logger.info(f"   🥇 Best similarity: {results[0][1]:.4f}")
                logger.info(f"   📰 Sources found: {list(set(article.source for article, _ in results))}")
                for i, (article, similarity) in enumerate(results[:3], 1):
                    logger.info(f"   #{i}: {article.title[:50]}... (score: {similarity:.4f}, source: {article.source})")
            
            return results
            
        except Exception as e:
            logger.error(f"Error finding relevant articles: {e}")
            return []
    
    def get_articles_for_fact_check(self, claim_text: str, max_articles: int = 5) -> str:
        """Get formatted RSS articles for fact-checking."""
        relevant_articles = self.find_relevant_articles(claim_text, max_articles)
        
        if not relevant_articles:
            return "לא נמצאו מקורות RSS רלוונטיים."
        
        # Format articles for the LLM
        formatted_articles = []
        for i, (article, similarity) in enumerate(relevant_articles, 1):
            formatted_article = f"""
מקור RSS #{i} (רלוונטיות: {similarity:.3f}):
כותרת: {article.title}
מקור: {article.source}
קישור: {article.link}
תאריך פרסום: {article.published}
תוכן: {article.clean_text[:800]}...
---
"""
            formatted_articles.append(formatted_article)
        
        return "\n".join(formatted_articles)
    
    def get_stats(self) -> Dict:
        """Get statistics about the RSS articles database."""
        if not self.articles:
            return {"total_articles": 0, "sources": [], "date_range": "N/A"}
        
        sources = list(set(article.source for article in self.articles))
        years = [article.get_publication_year() for article in self.articles]
        valid_years = [year for year in years if year is not None]
        
        return {
            "total_articles": len(self.articles),
            "sources": sources,
            "date_range": f"{min(valid_years) if valid_years else 'N/A'} - {max(valid_years) if valid_years else 'N/A'}",
            "tfidf_features": self.tfidf_matrix.shape[1] if self.tfidf_matrix is not None else 0
        }

# Global instance
_rss_matcher = None

def get_rss_matcher() -> RSSTFIDFMatcher:
    """Get or create the global RSS TF-IDF matcher instance."""
    global _rss_matcher
    if _rss_matcher is None:
        _rss_matcher = RSSTFIDFMatcher()
    return _rss_matcher

def find_relevant_rss_articles(claim_text: str, max_articles: int = 5) -> str:
    """Find relevant RSS articles for a claim using TF-IDF similarity."""
    matcher = get_rss_matcher()
    return matcher.get_articles_for_fact_check(claim_text, max_articles)

if __name__ == "__main__":
    # Test the RSS matcher
    logging.basicConfig(level=logging.INFO)
    
    matcher = RSSTFIDFMatcher()
    stats = matcher.get_stats()
    print(f"RSS Matcher Stats: {stats}")
    
    # Test with a sample query
    test_query = "נתניהו טראמפ פגישה"
    results = matcher.find_relevant_articles(test_query, 3)
    
    print(f"\nTop 3 articles for '{test_query}':")
    for i, (article, similarity) in enumerate(results, 1):
        print(f"{i}. {article.title} (similarity: {similarity:.3f})")
        print(f"   Source: {article.source}")
        print(f"   Link: {article.link}")
        print() 