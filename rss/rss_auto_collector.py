#!/usr/bin/env python3
"""
RSS Auto Collector - Automatically collects and processes RSS feeds
"""

import asyncio
import json
import logging
import os
import sys
import threading
import time
from datetime import datetime, timedelta, timezone
from hashlib import md5
from pathlib import Path
from typing import Any, Dict, List

import feedparser
import schedule

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# from rss.rss_processor import process_rss_with_gemini  # Not needed for basic collection
from simple_similarity import SentenceTransformer

# Import data persistence manager
try:
    from data_persistence import auto_backup_if_needed, create_data_backup
except ImportError:
    # Fallback if data_persistence module is not available
    def auto_backup_if_needed() -> None:
        pass
    def create_data_backup() -> str:
        return ""

# Configure logging with clean output
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Suppress noisy loggers
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("feedparser").setLevel(logging.WARNING)
logging.getLogger("asyncio").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

# RSS Feed URLs - Israeli and International News
RSS_FEEDS = {
    # Israeli News (Hebrew)
    "ynet": "https://www.ynet.co.il/Integration/StoryRss2.xml",
    "mako": "https://rcs.mako.co.il/rss/news-israel.xml",
    "israel_hayom": "https://www.israelhayom.co.il/rss.xml",
    "haaretz": "https://www.haaretz.co.il/cmlink/1.628180?fmt=rss",
    "globes": "https://www.globes.co.il/webservice/allrssmainnews.aspx?preflang=he",
    "walla": "https://rss.walla.co.il/feed/22?type=main",
    "kan": "https://www.kan.org.il/rss/news.xml",
    "galatz": "https://www.glz.co.il/rss.aspx",
    "n12": "https://www.mako.co.il/rss/news-israel.xml",
    "ch13": "https://13news.co.il/rss.xml",
    "inn": "https://www.inn.co.il/Rss.aspx",
    "maariv": "https://www.maariv.co.il/Rss/RssFeedsAllArticles.aspx",
    "nrg": "https://www.nrg.co.il/rss/rssfeedsrss.aspx",
    
    # International News (English)
    "bbc": "http://feeds.bbci.co.uk/news/rss.xml",
    "cnn": "http://rss.cnn.com/rss/edition.rss",
    "al_jazeera": "https://www.aljazeera.com/xml/rss/all.xml",
    "nyt": "https://rss.nytimes.com/services/xml/rss/nyt/HomePage.xml",
    "guardian": "https://www.theguardian.com/world/rss",
    "dw": "https://rss.dw.com/rdf/rss-en-all",
    "ap": "https://apnews.com/rss",
    "jerusalem_post": "https://www.jpost.com/Rss/RSSFeedsHeadlines.aspx",
    "google_news": "https://news.google.com/rss?hl=en-US&gl=US&ceid=US:en",
    
    # Middle East (Arabic/English)
    "al_jazeera_ar": "https://www.aljazeera.net/aljazeerarss/araba.xml",
    "middle_east_eye": "https://www.middleeasteye.net/rss",
}

# Configuration
DATA_DIR = Path("data")
RSS_ARTICLES_FILE = DATA_DIR / "rss_articles.json"
VECTORS_DIR = DATA_DIR / "vectors"
UPDATE_INTERVAL_HOURS = 1  # Update every 1 hour (changed from 6)
MAX_ARTICLES_PER_FEED = 100  # Increased from 50 to keep more articles
RETENTION_DAYS = 3  # Reduced from 30 to 3 days to optimize memory usage


class RSSAutoCollector:
    """Automatic RSS collection and processing system."""
    
    def __init__(self):
        self.model = None
        self.is_running = False
        self.last_update = None
        self.last_cleanup = None
        
        # Ensure directories exist
        DATA_DIR.mkdir(exist_ok=True)
        VECTORS_DIR.mkdir(exist_ok=True)
        
    def initialize_model(self):
        """Initialize the sentence transformer model."""
        if self.model is None:
            logger.info("Initializing SentenceTransformer model...")
            self.model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
            logger.info("Model initialized successfully")
    
    def collect_single_feed(self, name: str, url: str) -> List[Dict[str, Any]]:
        """Collect articles from a single RSS feed."""
        try:
            logger.info(f"Collecting from {name}: {url}")
            feed = feedparser.parse(url)
            
            if feed.bozo:
                logger.warning(f"Feed {name} has parsing issues: {feed.bozo_exception}")
            
            articles = []
            for entry in feed.entries[:MAX_ARTICLES_PER_FEED]:
                try:
                    # Handle missing fields gracefully
                    article = {
                        "title": entry.get("title", None),
                        "link": entry.get("link", None),
                        "published": entry.get("published", None),
                        "summary": entry.get("summary", None),
                        "content": self._extract_content(entry),
                        "source": name,
                        "collected_at": datetime.now(timezone.utc).isoformat()
                    }
                    
                    # Only require link field, make others optional
                    if article["link"]:
                        # Convert None values to empty strings for consistency
                        for key in ["title", "summary", "content"]:
                            if article[key] is None:
                                article[key] = ""
                        articles.append(article)
                        
                except Exception as e:
                    logger.warning(f"Error processing entry from {name}: {e}")
                    continue
            
            logger.info(f"Collected {len(articles)} articles from {name}")
            return articles
            
        except Exception as e:
            logger.error(f"Error collecting from {name} ({url}): {e}")
            return []
    
    def _extract_content(self, entry) -> str:
        """Extract content from RSS entry."""
        # Try different content fields
        if hasattr(entry, 'content') and entry.content:
            return entry.content[0].get('value', '')
        elif hasattr(entry, 'description'):
            return entry.description
        elif hasattr(entry, 'summary'):
            return entry.summary
        return ""
    
    async def collect_all_feeds(self) -> List[Dict[str, Any]]:
        """Collect articles from all RSS feeds concurrently."""
        logger.info(f"Starting collection from {len(RSS_FEEDS)} RSS feeds...")
        
        # Create tasks for concurrent collection
        tasks = []
        for name, url in RSS_FEEDS.items():
            task = asyncio.create_task(
                asyncio.to_thread(self.collect_single_feed, name, url)
            )
            tasks.append(task)
        
        # Wait for all collections to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Combine all articles
        all_articles = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                feed_name = list(RSS_FEEDS.keys())[i]
                logger.error(f"Failed to collect from {feed_name}: {result}")
            elif isinstance(result, list):
                all_articles.extend(result)
        
        logger.info(f"Total articles collected: {len(all_articles)}")
        return all_articles
    
    def _is_article_expired(self, article: Dict[str, Any]) -> bool:
        """Check if an article is older than the retention period."""
        try:
            collected_at_str = article.get('collected_at', '')
            if not collected_at_str:
                return True  # Remove articles without collection timestamp
            
            collected_at = datetime.fromisoformat(collected_at_str.replace('Z', '+00:00'))
            cutoff_date = datetime.now(timezone.utc) - timedelta(days=RETENTION_DAYS)
            
            return collected_at < cutoff_date
        except Exception as e:
            logger.warning(f"Error checking article expiration: {e}")
            return True  # Remove articles with invalid timestamps
    
    def cleanup_old_articles(self) -> int:
        """Remove articles older than retention period."""
        try:
            if not RSS_ARTICLES_FILE.exists():
                return 0
            
            with open(RSS_ARTICLES_FILE, 'r', encoding='utf-8') as f:
                articles = json.load(f)
            
            original_count = len(articles)
            
            # Filter out expired articles
            valid_articles = [
                article for article in articles 
                if not self._is_article_expired(article)
            ]
            
            removed_count = original_count - len(valid_articles)
            
            if removed_count > 0:
                # Save the filtered articles
                with open(RSS_ARTICLES_FILE, 'w', encoding='utf-8') as f:
                    json.dump(valid_articles, f, ensure_ascii=False, indent=2)
                
                logger.info(f"Cleaned up {removed_count} expired articles, "
                           f"kept {len(valid_articles)} articles")
            
            return removed_count
            
        except Exception as e:
            logger.error(f"Error cleaning up old articles: {e}")
            return 0
    
    def cleanup_old_vectors(self) -> int:
        """Remove vector files for articles that no longer exist or are expired."""
        try:
            if not RSS_ARTICLES_FILE.exists():
                return 0
            
            # Load current articles to get valid article IDs
            with open(RSS_ARTICLES_FILE, 'r', encoding='utf-8') as f:
                articles = json.load(f)
            
            # Create set of valid article IDs
            valid_article_ids = set()
            for article in articles:
                link = article.get('link', '')
                if link:
                    article_id = md5(link.encode('utf-8')).hexdigest()
                    valid_article_ids.add(article_id)
            
            # Find all vector files
            vector_files = list(VECTORS_DIR.glob("*.json"))
            removed_count = 0
            
            for vector_file in vector_files:
                try:
                    # Extract article ID from filename
                    filename = vector_file.name
                    if '_title.json' in filename:
                        article_id = filename.replace('_title.json', '')
                    elif '_text.json' in filename:
                        article_id = filename.replace('_text.json', '')
                    else:
                        continue
                    
                    # Remove if article ID is not in valid set
                    if article_id not in valid_article_ids:
                        vector_file.unlink()
                        removed_count += 1
                        
                except Exception as e:
                    logger.warning(f"Error processing vector file {vector_file}: {e}")
                    continue
            
            if removed_count > 0:
                logger.info(f"Cleaned up {removed_count} orphaned vector files")
            
            return removed_count
            
        except Exception as e:
            logger.error(f"Error cleaning up old vectors: {e}")
            return 0
    
    async def initialize_with_existing_data(self):
        """Initialize vectors for existing articles that don't have vectors yet."""
        try:
            if not RSS_ARTICLES_FILE.exists():
                logger.info("No existing articles file found")
                return
            
            logger.info("Initializing vectors for existing articles...")
            
            with open(RSS_ARTICLES_FILE, 'r', encoding='utf-8') as f:
                articles = json.load(f)
            
            # Filter articles that need vectors
            articles_needing_vectors = []
            for article in articles:
                if self._is_article_expired(article):
                    continue  # Skip expired articles
                
                link = article.get('link', '')
                if not link:
                    continue
                
                article_id = md5(link.encode('utf-8')).hexdigest()
                title_vector_path = VECTORS_DIR / f"{article_id}_title.json"
                text_vector_path = VECTORS_DIR / f"{article_id}_text.json"
                
                # Add if either vector is missing
                if not title_vector_path.exists() or not text_vector_path.exists():
                    articles_needing_vectors.append(article)
            
            if articles_needing_vectors:
                logger.info(f"Found {len(articles_needing_vectors)} articles needing vectors")
                await self.generate_vectors(articles_needing_vectors)
            else:
                logger.info("All existing articles already have vectors")
                
        except Exception as e:
            logger.error(f"Error initializing existing data: {e}")
    
    def save_articles(self, articles: List[Dict[str, Any]]):
        """Save articles to JSON file with retention policy."""
        try:
            # Load existing articles if file exists
            existing_articles = []
            if RSS_ARTICLES_FILE.exists():
                with open(RSS_ARTICLES_FILE, 'r', encoding='utf-8') as f:
                    existing_articles = json.load(f)
            
            # Create a set of existing links to avoid duplicates
            existing_links = {article.get('link') for article in existing_articles}
            
            # Add only new articles
            new_articles = []
            for article in articles:
                if article.get('link') not in existing_links:
                    new_articles.append(article)
            
            # Combine new and existing articles
            combined_articles = new_articles + existing_articles
            
            # Filter out expired articles
            valid_articles = [
                article for article in combined_articles 
                if not self._is_article_expired(article)
            ]
            
            # Sort by collection time (most recent first)
            valid_articles.sort(
                key=lambda x: x.get('collected_at', ''), 
                reverse=True
            )
            
            # Save to file
            with open(RSS_ARTICLES_FILE, 'w', encoding='utf-8') as f:
                json.dump(valid_articles, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Saved {len(new_articles)} new articles, "
                       f"total: {len(valid_articles)} articles (within {RETENTION_DAYS} days)")
            
            return len(new_articles)
            
        except Exception as e:
            logger.error(f"Error saving articles: {e}")
            return 0
    
    async def generate_vectors(self, articles: List[Dict[str, Any]]):
        """Generate TF-IDF vectors for new articles."""
        if not self.model:
            self.initialize_model()
        
        logger.info("Generating vectors for articles...")
        
        # Extract texts for TF-IDF corpus
        texts = []
        for article in articles:
            title = article.get('title', '')
            content = article.get('content', '') or article.get('summary', '')
            if title or content:
                texts.append(f"{title} {content}")
        
        # Add texts to corpus and generate vectors
        self.model.add_to_corpus(texts)
        
        # Save vectors for each article
        for article in articles:
            try:
                link = article.get('link', '')
                if not link:
                    continue
                
                # Create unique ID for the article
                article_id = md5(link.encode('utf-8')).hexdigest()
                
                # Generate combined vector for title and content
                title = article.get('title', '')
                content = article.get('content', '') or article.get('summary', '')
                combined_text = f"{title} {content}"
                
                if combined_text:
                    vector = self.model.encode(combined_text)
                    vector_path = VECTORS_DIR / f"{article_id}.json"
                    with open(vector_path, 'w') as f:
                        json.dump(vector.tolist(), f)
                
            except Exception as e:
                logger.warning(f"Error generating vector for article: {e}")
                continue
        
        logger.info(f"Generated vectors for {len(articles)} articles")
    
    async def perform_cleanup(self):
        """Perform cleanup of old articles and vectors."""
        try:
            logger.info("Starting cleanup of old data...")
            
            # Clean up old articles
            removed_articles = self.cleanup_old_articles()
            
            # Clean up orphaned vectors
            removed_vectors = self.cleanup_old_vectors()
            
            self.last_cleanup = datetime.now(timezone.utc)
            
            logger.info(f"Cleanup completed: removed {removed_articles} articles "
                       f"and {removed_vectors} vector files")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
    
    async def update_rss_data(self):
        """Main update function - collect feeds, generate vectors, and cleanup."""
        try:
            logger.info("Starting RSS data update...")
            self.is_running = True
            
            # Log current status before update
            current_status = self.get_status()
            logger.info(f"📊 RSS Status BEFORE update:")
            logger.info(f"   📰 Current articles: {current_status['current_articles']}")
            logger.info(f"   🔢 Current vectors: {current_status['current_vectors']}")
            logger.info(f"   📡 Total RSS feeds: {current_status['total_feeds']}")
            logger.info(f"   📅 Retention period: {current_status['retention_days']} days")
            
            # Create backup if needed (before making changes)
            try:
                auto_backup_if_needed()
            except Exception as e:
                logger.warning(f"Backup check failed: {e}")
            
            # Collect articles from all feeds
            articles = await self.collect_all_feeds()
            
            if articles:
                # Save articles (with retention policy)
                saved_count = self.save_articles(articles)
                
                # Generate vectors for new articles
                await self.generate_vectors(articles)
                
                logger.info(f"✅ RSS Update Summary:")
                logger.info(f"   📥 Collected: {len(articles)} articles")
                logger.info(f"   💾 Saved: {saved_count} new articles")
                
                # Create backup after successful update if significant changes
                if saved_count > 10:  # Only backup if we added significant new content
                    try:
                        backup_path = create_data_backup()
                        if backup_path:
                            logger.info(f"📦 Created backup after update: {Path(backup_path).name}")
                    except Exception as e:
                        logger.warning(f"Post-update backup failed: {e}")
            else:
                logger.warning("❌ No articles collected during update")
            
            # Perform cleanup (remove old articles and vectors)
            await self.perform_cleanup()
            
            # Log final status after update
            final_status = self.get_status()
            logger.info(f"📊 RSS Status AFTER update:")
            logger.info(f"   📰 Total articles: {final_status['current_articles']}")
            logger.info(f"   🔢 Total vectors: {final_status['current_vectors']}")
            
            self.last_update = datetime.now(timezone.utc)
            logger.info(f"RSS update completed successfully at {self.last_update}")
                
        except Exception as e:
            logger.error(f"Error during RSS update: {e}")
        finally:
            self.is_running = False
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status of the RSS collector."""
        # Count current articles and vectors
        article_count = 0
        if RSS_ARTICLES_FILE.exists():
            try:
                with open(RSS_ARTICLES_FILE, 'r', encoding='utf-8') as f:
                    articles = json.load(f)
                    article_count = len(articles)
            except Exception:
                pass
        
        vector_count = len(list(VECTORS_DIR.glob("*.json"))) if VECTORS_DIR.exists() else 0
        
        return {
            "is_running": self.is_running,
            "last_update": self.last_update.isoformat() if self.last_update else None,
            "last_cleanup": self.last_cleanup.isoformat() if self.last_cleanup else None,
            "total_feeds": len(RSS_FEEDS),
            "retention_days": RETENTION_DAYS,
            "current_articles": article_count,
            "current_vectors": vector_count,
            "articles_file_exists": RSS_ARTICLES_FILE.exists(),
            "vectors_dir_exists": VECTORS_DIR.exists(),
        }


# Global collector instance
collector = RSSAutoCollector()


def run_update():
    """Run RSS update (synchronous wrapper for async function)."""
    asyncio.run(collector.update_rss_data())


def run_initialization():
    """Run initialization with existing data (synchronous wrapper)."""
    asyncio.run(collector.initialize_with_existing_data())


def schedule_updates():
    """Schedule periodic RSS updates."""
    logger.info(f"Scheduling RSS updates every {UPDATE_INTERVAL_HOURS} hours")
    
    # Schedule updates
    schedule.every(UPDATE_INTERVAL_HOURS).hours.do(run_update)
    
    # Don't run initial update here - it will be handled by the async startup
    logger.info("RSS scheduler started - initial update will be handled separately")
    
    # Keep the scheduler running
    while True:
        schedule.run_pending()
        time.sleep(60)  # Check every minute


def start_background_scheduler():
    """Start the RSS update scheduler in a background thread."""
    # Note: This function should not run initialization directly when called from FastAPI startup
    # The initialization should be handled separately in the API startup
    scheduler_thread = threading.Thread(target=schedule_updates, daemon=True)
    scheduler_thread.start()
    logger.info("RSS auto-collector started in background")


async def start_background_scheduler_async():
    """Start the RSS update scheduler with async initialization."""
    # First, initialize with existing data
    logger.info("Initializing with existing data...")
    await collector.initialize_with_existing_data()
    
    # Run initial update
    logger.info("Running initial RSS update...")
    await collector.update_rss_data()
    
    # Then start the scheduler (without initial update)
    scheduler_thread = threading.Thread(target=schedule_updates, daemon=True)
    scheduler_thread.start()
    logger.info("RSS auto-collector started in background")


async def manual_update():
    """Manually trigger an RSS update."""
    await collector.update_rss_data()


async def manual_initialization():
    """Manually trigger initialization with existing data."""
    await collector.initialize_with_existing_data()


async def manual_cleanup():
    """Manually trigger cleanup of old data."""
    await collector.perform_cleanup()


def get_collector_status():
    """Get the current status of the RSS collector."""
    return collector.get_status()


# Main function for standalone execution
def main():
    """Main function for running the RSS collector."""
    import argparse
    
    parser = argparse.ArgumentParser(description="RSS Auto Collector")
    parser.add_argument("--update", action="store_true", help="Run a single update")
    parser.add_argument("--schedule", action="store_true", help="Start scheduled updates")
    parser.add_argument("--status", action="store_true", help="Show status")
    parser.add_argument("--init", action="store_true", help="Initialize with existing data")
    parser.add_argument("--cleanup", action="store_true", help="Clean up old data")
    
    args = parser.parse_args()
    
    if args.update:
        logger.info("Running manual RSS update...")
        run_update()
    elif args.schedule:
        logger.info("Starting RSS scheduler...")
        schedule_updates()
    elif args.status:
        status = get_collector_status()
        print(json.dumps(status, indent=2))
    elif args.init:
        logger.info("Initializing with existing data...")
        run_initialization()
    elif args.cleanup:
        logger.info("Running cleanup...")
        asyncio.run(collector.perform_cleanup())
    else:
        logger.info("No action specified. Use --help for options.")


if __name__ == "__main__":
    main() 