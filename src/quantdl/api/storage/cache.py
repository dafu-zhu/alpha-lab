"""Disk cache with LRU eviction and TTL support."""

import contextlib
import json
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import polars as pl

from quantdl.api.exceptions import CacheError


@dataclass
class CacheEntry:
    """Metadata for a cached file."""

    local_path: str
    size_bytes: int
    fetched_at: float
    last_accessed: float
    storage_path: str


class DiskCache:
    """Thread-safe disk cache with LRU eviction and TTL.

    Cache structure:
        ~/.quantdl/cache/
        ├── metadata.json       # Entry metadata
        └── data/               # Cached parquet files
            └── {hash}.parquet
    """

    DEFAULT_CACHE_DIR = "~/.quantdl/cache"
    DEFAULT_TTL_SECONDS = 24 * 60 * 60  # 24 hours
    DEFAULT_MAX_SIZE_BYTES = 10 * 1024 * 1024 * 1024  # 10 GB

    def __init__(
        self,
        cache_dir: str | None = None,
        ttl_seconds: int | None = None,
        max_size_bytes: int | None = None,
    ) -> None:
        self._cache_dir = Path(cache_dir or self.DEFAULT_CACHE_DIR).expanduser()
        self._data_dir = self._cache_dir / "data"
        self._metadata_path = self._cache_dir / "metadata.json"
        self._ttl = ttl_seconds if ttl_seconds is not None else self.DEFAULT_TTL_SECONDS
        self._max_size = max_size_bytes if max_size_bytes is not None else self.DEFAULT_MAX_SIZE_BYTES
        self._lock = threading.RLock()
        self._entries: dict[str, CacheEntry] = {}

        self._ensure_dirs()
        self._load_metadata()

    def _ensure_dirs(self) -> None:
        """Create cache directories if they don't exist."""
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._data_dir.mkdir(parents=True, exist_ok=True)

    def _load_metadata(self) -> None:
        """Load cache metadata from disk."""
        with self._lock:
            if self._metadata_path.exists():
                try:
                    with open(self._metadata_path) as f:
                        data = json.load(f)
                    self._entries = {
                        k: CacheEntry(**v) for k, v in data.items()
                    }
                except (json.JSONDecodeError, TypeError, KeyError) as e:
                    # Corrupted metadata, reset
                    self._entries = {}
                    self._save_metadata()
                    raise CacheError(f"Corrupted cache metadata, reset: {e}") from e

    def _save_metadata(self) -> None:
        """Persist cache metadata to disk atomically."""
        with self._lock:
            temp_path = self._metadata_path.with_suffix(".tmp")
            with open(temp_path, "w") as f:
                json.dump({k: asdict(v) for k, v in self._entries.items()}, f, indent=2)
            temp_path.replace(self._metadata_path)

    def _hash_path(self, storage_path: str) -> str:
        """Generate a safe filename from storage path."""
        import hashlib
        return hashlib.sha256(storage_path.encode()).hexdigest()[:16]

    def _total_size(self) -> int:
        """Calculate total cache size in bytes."""
        return sum(e.size_bytes for e in self._entries.values())

    def _evict_lru(self, needed_bytes: int = 0) -> None:
        """Evict least recently used entries until under size limit."""
        with self._lock:
            while self._total_size() + needed_bytes > self._max_size and self._entries:
                # Find LRU entry
                lru_key = min(self._entries, key=lambda k: self._entries[k].last_accessed)
                entry = self._entries.pop(lru_key)
                with contextlib.suppress(OSError):
                    Path(entry.local_path).unlink(missing_ok=True)
            self._save_metadata()

    def _evict_expired(self) -> None:
        """Remove entries that have exceeded TTL."""
        now = time.time()
        with self._lock:
            expired = [
                k for k, e in self._entries.items()
                if now - e.fetched_at > self._ttl
            ]
            for key in expired:
                entry = self._entries.pop(key)
                with contextlib.suppress(OSError):
                    Path(entry.local_path).unlink(missing_ok=True)
            if expired:
                self._save_metadata()

    def get(self, storage_path: str) -> pl.DataFrame | None:
        """Get cached DataFrame if valid.

        Args:
            storage_path: The storage path used as cache key

        Returns:
            DataFrame if cache hit and valid, None otherwise
        """
        with self._lock:
            self._evict_expired()

            if storage_path not in self._entries:
                return None

            entry = self._entries[storage_path]
            local_path = Path(entry.local_path)

            if not local_path.exists():
                # File was deleted externally
                del self._entries[storage_path]
                self._save_metadata()
                return None

            # Update last accessed time
            entry.last_accessed = time.time()
            self._entries[storage_path] = CacheEntry(
                local_path=entry.local_path,
                size_bytes=entry.size_bytes,
                fetched_at=entry.fetched_at,
                last_accessed=entry.last_accessed,
                storage_path=entry.storage_path,
            )
            self._save_metadata()

            try:
                return pl.read_parquet(local_path)
            except Exception:
                # Corrupted file, remove from cache
                del self._entries[storage_path]
                local_path.unlink(missing_ok=True)
                self._save_metadata()
                return None

    def put(self, storage_path: str, df: pl.DataFrame) -> None:
        """Store DataFrame in cache.

        Args:
            storage_path: The storage path to use as cache key
            df: DataFrame to cache
        """
        with self._lock:
            file_hash = self._hash_path(storage_path)
            local_path = self._data_dir / f"{file_hash}.parquet"

            # Write to temp file first (atomic)
            temp_path = local_path.with_suffix(".tmp")
            df.write_parquet(temp_path)
            size_bytes = temp_path.stat().st_size

            # Evict if needed before moving file
            self._evict_lru(size_bytes)

            temp_path.replace(local_path)

            now = time.time()
            self._entries[storage_path] = CacheEntry(
                local_path=str(local_path),
                size_bytes=size_bytes,
                fetched_at=now,
                last_accessed=now,
                storage_path=storage_path,
            )
            self._save_metadata()

    def invalidate(self, storage_path: str) -> bool:
        """Remove specific entry from cache.

        Returns:
            True if entry existed and was removed
        """
        with self._lock:
            if storage_path not in self._entries:
                return False

            entry = self._entries.pop(storage_path)
            Path(entry.local_path).unlink(missing_ok=True)
            self._save_metadata()
            return True

    def clear(self) -> None:
        """Remove all cached data."""
        with self._lock:
            for entry in self._entries.values():
                Path(entry.local_path).unlink(missing_ok=True)
            self._entries.clear()
            self._save_metadata()

    def stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            return {
                "entries": len(self._entries),
                "total_size_bytes": self._total_size(),
                "max_size_bytes": self._max_size,
                "ttl_seconds": self._ttl,
                "cache_dir": str(self._cache_dir),
            }
