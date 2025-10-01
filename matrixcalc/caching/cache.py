"""Caching system with LRU policy and TTL support."""

import time
from typing import Any, Dict, Optional, Tuple
from collections import OrderedDict
import threading

from ..logging.setup import get_logger

logger = get_logger(__name__)


class LRUCache:
    """LRU Cache with TTL support."""
    
    def __init__(self, max_size: int = 100, ttl_seconds: int = 3600):
        """
        Initialize LRU cache.
        
        Args:
            max_size: Maximum number of items to cache
            ttl_seconds: Time-to-live for cached items in seconds
        """
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache: OrderedDict[str, Tuple[Any, float]] = OrderedDict()
        self._lock = threading.RLock()
    
    def _is_expired(self, timestamp: float) -> bool:
        """Check if an item is expired."""
        return time.time() - timestamp > self.ttl_seconds
    
    def _cleanup_expired(self) -> None:
        """Remove expired items from cache."""
        current_time = time.time()
        expired_keys = [
            key for key, (_, timestamp) in self._cache.items()
            if current_time - timestamp > self.ttl_seconds
        ]
        
        for key in expired_keys:
            del self._cache[key]
            logger.debug(f"Removed expired cache entry: {key}")
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache."""
        with self._lock:
            if key not in self._cache:
                return None
            
            value, timestamp = self._cache[key]
            
            if self._is_expired(timestamp):
                del self._cache[key]
                logger.debug(f"Cache miss (expired): {key}")
                return None
            
            # Move to end (most recently used)
            self._cache.move_to_end(key)
            logger.debug(f"Cache hit: {key}")
            return value
    
    def put(self, key: str, value: Any) -> None:
        """Put item in cache."""
        with self._lock:
            current_time = time.time()
            
            # If key exists, update it
            if key in self._cache:
                self._cache[key] = (value, current_time)
                self._cache.move_to_end(key)
                logger.debug(f"Updated cache entry: {key}")
                return
            
            # If cache is full, remove least recently used
            if len(self._cache) >= self.max_size:
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]
                logger.debug(f"Evicted cache entry: {oldest_key}")
            
            self._cache[key] = (value, current_time)
            logger.debug(f"Added cache entry: {key}")
    
    def clear(self) -> None:
        """Clear all items from cache."""
        with self._lock:
            self._cache.clear()
            logger.debug("Cache cleared")
    
    def size(self) -> int:
        """Get current cache size."""
        with self._lock:
            self._cleanup_expired()
            return len(self._cache)
    
    def keys(self) -> list[str]:
        """Get all non-expired cache keys."""
        with self._lock:
            self._cleanup_expired()
            return list(self._cache.keys())


class CacheManager:
    """Manages multiple caches for different operations."""
    
    def __init__(self):
        self._caches: Dict[str, LRUCache] = {}
        self._lock = threading.RLock()
    
    def get_cache(self, name: str, max_size: int = 100, ttl_seconds: int = 3600) -> LRUCache:
        """Get or create a cache with the given name."""
        with self._lock:
            if name not in self._caches:
                self._caches[name] = LRUCache(max_size, ttl_seconds)
                logger.info(f"Created cache '{name}' with size {max_size}, TTL {ttl_seconds}s")
            return self._caches[name]
    
    def clear_cache(self, name: str) -> None:
        """Clear a specific cache."""
        with self._lock:
            if name in self._caches:
                self._caches[name].clear()
                logger.info(f"Cleared cache '{name}'")
    
    def clear_all_caches(self) -> None:
        """Clear all caches."""
        with self._lock:
            for cache in self._caches.values():
                cache.clear()
            logger.info("Cleared all caches")
    
    def get_cache_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all caches."""
        with self._lock:
            stats = {}
            for name, cache in self._caches.items():
                stats[name] = {
                    'size': cache.size(),
                    'max_size': cache.max_size,
                    'ttl_seconds': cache.ttl_seconds
                }
            return stats


# Global cache manager instance
cache_manager = CacheManager()
