"""
cv_agent.cache
==============
Thread-safe LRU cache with per-entry TTL.

Used for caching CV candidates and judge ensemble results to avoid
redundant GPU inference calls.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from typing import Any, Dict, Optional

from cv_agent.config import PipelineConfig, logger


# ==============================================================================
# CACHE ENTRY
# ==============================================================================

class _CacheEntry:
    __slots__ = ("value", "expires_at")

    def __init__(self, value: Any, ttl: int) -> None:
        self.value      = value
        self.expires_at = time.monotonic() + ttl


# ==============================================================================
# LRU CACHE
# ==============================================================================

class LRUCache:
    """
    Thread-safe LRU cache with per-entry TTL.

    Cache key schema: <profile_hash>:<jd_hash>:<iteration>:<label>
    Separate namespaces for candidates ("cv") and judge outputs ("judge").
    """

    def __init__(self, max_size: int = 256, ttl: int = 3600) -> None:
        self._max_size = max_size
        self._ttl      = ttl
        self._cache: OrderedDict[str, _CacheEntry] = OrderedDict()
        self._lock  = threading.Lock()
        self._hits  = 0
        self._misses = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @staticmethod
    def make_key(
        profile_hash: str,
        jd_hash: str,
        iteration: int,
        label: str,
        namespace: str = "cv",
        hallucination_hash: str = "0" * 8,
    ) -> str:
        # hallucination_hash ensures hallucinated outputs are never reused after
        # corrections — any change in the issues list produces a distinct key.
        return f"{namespace}:{profile_hash}:{jd_hash}:{iteration}:{label}:{hallucination_hash}"

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                self._misses += 1
                return None
            if time.monotonic() > entry.expires_at:
                del self._cache[key]
                self._misses += 1
                return None
            # Move to end (most recently used)
            self._cache.move_to_end(key)
            self._hits += 1
            return entry.value

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            self._cache[key] = _CacheEntry(value, self._ttl)
            while len(self._cache) > self._max_size:
                self._cache.popitem(last=False)  # evict LRU

    def stats(self) -> Dict[str, int]:
        with self._lock:
            return {
                "size":   len(self._cache),
                "hits":   self._hits,
                "misses": self._misses,
            }

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()


# ==============================================================================
# GLOBAL CACHE — singleton with double-checked locking
# ==============================================================================

_cache: Optional[LRUCache] = None
_cache_init_lock = threading.Lock()   # FIX C-2: prevent double-init race condition


def get_cache(cfg: Optional[PipelineConfig] = None) -> LRUCache:
    """
    Return (or lazily create) the process-global LRU cache.

    FIX C-2 (v7): double-checked locking guarantees exactly one
    initialisation across all threads.
    """
    global _cache
    if _cache is None:
        with _cache_init_lock:
            if _cache is None:          # second check inside the lock
                size   = cfg.cache_max_size    if cfg else 256
                ttl    = cfg.cache_ttl_seconds if cfg else 3600
                _cache = LRUCache(max_size=size, ttl=ttl)
                logger.info("LRUCache initialised (max_size=%d, ttl=%ds)", size, ttl)
    return _cache
