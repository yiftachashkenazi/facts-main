#!/usr/bin/env python3
"""
Data Persistence Manager - Ensures RSS data survives deployments
"""

import json
import logging
import os
import shutil
import tarfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Configuration
DATA_DIR = Path("data")
BACKUP_DIR = Path("data_backups")
RSS_ARTICLES_FILE = DATA_DIR / "rss_articles.json"
VECTORS_DIR = DATA_DIR / "vectors"

# Backup settings
MAX_BACKUPS = 10  # Keep last 10 backups
BACKUP_INTERVAL_HOURS = 6  # Create backup every 6 hours


class DataPersistenceManager:
    """
    Manages data persistence and backups for RSS system.
    This class handles the creation, restoration, and cleanup of backups,
    as well as providing statistics about the current data state.
    It ensures that critical data like RSS articles and vector embeddings
    are preserved across deployments or system restarts.
    """
    
    def __init__(self):
        self.ensure_directories()
        
    def ensure_directories(self):
        """Ensure all necessary directories exist."""
        DATA_DIR.mkdir(exist_ok=True)
        VECTORS_DIR.mkdir(exist_ok=True)
        BACKUP_DIR.mkdir(exist_ok=True)
        
    def create_backup(self) -> str:
        """Create a compressed backup of all RSS data."""
        try:
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            backup_filename = f"rss_backup_{timestamp}.tar.gz"
            backup_path = BACKUP_DIR / backup_filename
            
            logger.info(f"Creating RSS data backup: {backup_filename}")
            
            # Create tar.gz backup
            with tarfile.open(backup_path, "w:gz") as tar:
                # Add RSS articles file if it exists
                if RSS_ARTICLES_FILE.exists():
                    tar.add(RSS_ARTICLES_FILE, arcname="rss_articles.json")
                    logger.info(f"Added RSS articles file to backup ({RSS_ARTICLES_FILE.stat().st_size} bytes)")
                
                # Add vectors directory if it exists
                if VECTORS_DIR.exists():
                    tar.add(VECTORS_DIR, arcname="vectors")
                    vector_count = len(list(VECTORS_DIR.glob("*.json")))
                    logger.info(f"Added {vector_count} vector files to backup")
            
            # Get backup size
            backup_size = backup_path.stat().st_size
            logger.info(f"✅ Backup created successfully: {backup_filename} ({backup_size:,} bytes)")
            
            # Clean up old backups
            self.cleanup_old_backups()
            
            return str(backup_path)
            
        except Exception as e:
            logger.error(f"Error creating backup: {e}")
            return ""
    
    def restore_from_backup(self, backup_path: Optional[str] = None) -> bool:
        """Restore RSS data from backup."""
        try:
            if backup_path is None:
                # Find the most recent backup
                backup_files = list(BACKUP_DIR.glob("rss_backup_*.tar.gz"))
                if not backup_files:
                    logger.warning("No backup files found")
                    return False
                
                backup_file = max(backup_files, key=lambda x: x.stat().st_mtime)
                logger.info(f"Using most recent backup: {backup_file.name}")
            else:
                backup_file = Path(backup_path)
                if not backup_file.exists():
                    logger.error(f"Backup file not found: {backup_path}")
                    return False
            
            logger.info(f"Restoring RSS data from backup: {backup_file.name}")
            
            # Extract backup
            with tarfile.open(backup_file, "r:gz") as tar:
                tar.extractall(path=DATA_DIR.parent)
            
            # Verify restoration
            articles_restored = RSS_ARTICLES_FILE.exists()
            vectors_restored = VECTORS_DIR.exists()
            
            if articles_restored:
                with open(RSS_ARTICLES_FILE, 'r', encoding='utf-8') as f:
                    articles = json.load(f)
                logger.info(f"✅ Restored {len(articles)} RSS articles")
            
            if vectors_restored:
                vector_count = len(list(VECTORS_DIR.glob("*.json")))
                logger.info(f"✅ Restored {vector_count} vector files")
            
            logger.info("RSS data restoration completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error restoring from backup: {e}")
            return False
    
    def cleanup_old_backups(self):
        """Remove old backup files, keeping only the most recent ones."""
        try:
            backup_files = list(BACKUP_DIR.glob("rss_backup_*.tar.gz"))
            if len(backup_files) <= MAX_BACKUPS:
                return
            
            # Sort by modification time (oldest first)
            backup_files.sort(key=lambda x: x.stat().st_mtime)
            
            # Remove oldest backups
            files_to_remove = backup_files[:-MAX_BACKUPS]
            for backup_file in files_to_remove:
                backup_file.unlink()
                logger.info(f"Removed old backup: {backup_file.name}")
            
            logger.info(f"Cleaned up {len(files_to_remove)} old backup files")
            
        except Exception as e:
            logger.error(f"Error cleaning up old backups: {e}")
    
    def get_data_stats(self) -> Dict[str, Any]:
        """Get statistics about current RSS data."""
        stats: Dict[str, Any] = {
            "articles_file_exists": RSS_ARTICLES_FILE.exists(),
            "vectors_dir_exists": VECTORS_DIR.exists(),
            "articles_count": 0,
            "vectors_count": 0,
            "articles_file_size": 0,
            "vectors_dir_size": 0,
            "backup_count": 0,
            "latest_backup": None,
        }
        
        try:
            # RSS articles stats
            if RSS_ARTICLES_FILE.exists():
                stats["articles_file_size"] = RSS_ARTICLES_FILE.stat().st_size
                with open(RSS_ARTICLES_FILE, 'r', encoding='utf-8') as f:
                    articles = json.load(f)
                    stats["articles_count"] = len(articles)
            
            # Vectors stats
            if VECTORS_DIR.exists():
                vector_files = list(VECTORS_DIR.glob("*.json"))
                stats["vectors_count"] = len(vector_files)
                stats["vectors_dir_size"] = sum(f.stat().st_size for f in vector_files)
            
            # Backup stats
            backup_files = list(BACKUP_DIR.glob("rss_backup_*.tar.gz"))
            stats["backup_count"] = len(backup_files)
            if backup_files:
                latest_backup = max(backup_files, key=lambda x: x.stat().st_mtime)
                stats["latest_backup"] = {
                    "filename": latest_backup.name,
                    "size": latest_backup.stat().st_size,
                    "created": datetime.fromtimestamp(latest_backup.stat().st_mtime).isoformat()
                }
        
        except Exception as e:
            logger.error(f"Error getting data stats: {e}")
        
        return stats
    
    def should_create_backup(self) -> bool:
        """Check if it's time to create a new backup."""
        try:
            backup_files = list(BACKUP_DIR.glob("rss_backup_*.tar.gz"))
            if not backup_files:
                return True  # No backups exist
            
            # Check if enough time has passed since last backup
            latest_backup = max(backup_files, key=lambda x: x.stat().st_mtime)
            last_backup_time = latest_backup.stat().st_mtime
            current_time = time.time()
            
            hours_since_backup = (current_time - last_backup_time) / 3600
            return hours_since_backup >= BACKUP_INTERVAL_HOURS
            
        except Exception as e:
            logger.error(f"Error checking backup schedule: {e}")
            return False
    
    def auto_backup_if_needed(self):
        """Automatically create backup if needed."""
        if self.should_create_backup():
            self.create_backup()


# Global instance
persistence_manager = DataPersistenceManager()


def create_data_backup() -> str:
    """Create a backup of RSS data."""
    return persistence_manager.create_backup()


def restore_data_from_backup(backup_path: Optional[str] = None) -> bool:
    """Restore RSS data from backup."""
    return persistence_manager.restore_from_backup(backup_path)


def get_data_statistics() -> Dict[str, Any]:
    """Get RSS data statistics."""
    return persistence_manager.get_data_stats()


def auto_backup_if_needed():
    """Auto backup if needed."""
    persistence_manager.auto_backup_if_needed()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="RSS Data Persistence Manager")
    parser.add_argument("--backup", action="store_true", help="Create backup")
    parser.add_argument("--restore", type=str, help="Restore from backup file")
    parser.add_argument("--stats", action="store_true", help="Show data statistics")
    parser.add_argument("--cleanup", action="store_true", help="Cleanup old backups")
    
    args = parser.parse_args()
    
    if args.backup:
        backup_path = create_data_backup()
        print(f"Backup created: {backup_path}")
    elif args.restore:
        success = restore_data_from_backup(args.restore)
        print(f"Restore {'successful' if success else 'failed'}")
    elif args.stats:
        stats = get_data_statistics()
        print(json.dumps(stats, indent=2))
    elif args.cleanup:
        persistence_manager.cleanup_old_backups()
        print("Cleanup completed")
    else:
        print("No action specified. Use --help for options.") 