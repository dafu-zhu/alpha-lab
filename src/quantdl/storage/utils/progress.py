"""
Progress Tracker for Upload Operations
Tracks completed security_ids or symbols for resume capability
"""
import json
from datetime import datetime, timezone
from typing import Set, Union, Literal
from quantdl.storage.utils.exceptions import NoSuchKeyError

KeyType = Union[int, str]


class UploadProgressTracker:
    """
    Tracks completed security_ids or symbols for upload operations.

    Stores progress as JSON:
        data/upload_progress/{task_name}.json

    Schema:
        {
            'completed': [1001, 1002, ...] or ['AAPL', 'MSFT', ...],
            'last_updated': '2025-01-13T10:30:00Z',
            'stats': {'total': 5000, 'completed': 2500, 'failed': 10}
        }
    """

    def __init__(
        self,
        storage_client,
        task_name: str = 'daily_ticks_backfill',
        flush_interval: int = 100,
        key_type: Literal['int', 'str'] = 'int'
    ):
        """
        Initialize progress tracker.

        :param storage_client: LocalStorageClient instance
        :param task_name: Unique name for this upload task
        :param flush_interval: Save every N completions (default: 100)
        :param key_type: Type of keys to track ('int' for security_ids, 'str' for symbols)
        """
        self.storage_client = storage_client
        self.task_name = task_name
        self.key = f'data/upload_progress/{task_name}.json'
        self.flush_interval = flush_interval
        self.key_type = key_type

        self._completed: Set[KeyType] = set()
        self._pending_count = 0
        self._stats = {'total': 0, 'completed': 0, 'failed': 0, 'skipped': 0}
        self._loaded = False

    def load(self) -> Set[KeyType]:
        """
        Load completed keys from storage.

        :return: Set of completed keys (security_ids or symbols)
        """
        if self._loaded:
            return self._completed

        try:
            response = self.storage_client.get_object(
                Bucket='',
                Key=self.key
            )
            data = json.loads(response['Body'].read().decode('utf-8'))
            self._completed = set(data.get('completed', []))
            self._stats = data.get('stats', self._stats)
            self._loaded = True
        except NoSuchKeyError:
            # No progress file yet - start fresh
            self._completed = set()
            self._loaded = True

        return self._completed

    def mark_completed(self, key: KeyType):
        """
        Mark a key as completed.

        Automatically flushes every flush_interval completions.

        :param key: Key that was successfully processed (security_id or symbol)
        """
        if not self._loaded:
            self.load()

        self._completed.add(key)
        self._stats['completed'] = len(self._completed)
        self._pending_count += 1

        if self._pending_count >= self.flush_interval:
            self.save()
            self._pending_count = 0

    def mark_failed(self, key: KeyType):
        """
        Increment failed counter (does not mark as completed).

        :param key: Key that failed processing
        """
        self._stats['failed'] += 1

    def mark_skipped(self, key: KeyType):
        """
        Increment skipped counter.

        :param key: Key that was skipped
        """
        self._stats['skipped'] += 1

    def set_total(self, total: int):
        """
        Set total number of security_ids to process.

        :param total: Total count
        """
        self._stats['total'] = total

    def is_completed(self, key: KeyType) -> bool:
        """
        Check if a key has already been processed.

        :param key: Key to check (security_id or symbol)
        :return: True if already completed
        """
        if not self._loaded:
            self.load()
        return key in self._completed

    def get_pending(self, all_keys: Set[KeyType]) -> Set[KeyType]:
        """
        Get keys that haven't been processed yet.

        :param all_keys: Full set of keys to process
        :return: Set of pending keys
        """
        if not self._loaded:
            self.load()
        return all_keys - self._completed

    def save(self):
        """
        Save progress to storage.
        """
        data = {
            'completed': sorted(list(self._completed)),
            'last_updated': datetime.now(timezone.utc).isoformat(),
            'stats': self._stats
        }

        self.storage_client.put_object(
            Bucket='',
            Key=self.key,
            Body=json.dumps(data, indent=2).encode('utf-8'),
            ContentType='application/json'
        )

    def reset(self):
        """
        Reset progress (delete file and clear state).
        """
        try:
            self.storage_client.delete_object(
                Bucket='',
                Key=self.key
            )
        except Exception:
            pass

        self._completed: Set[KeyType] = set()
        self._stats = {'total': 0, 'completed': 0, 'failed': 0, 'skipped': 0}
        self._pending_count = 0
        self._loaded = True

    @property
    def completed(self) -> Set[KeyType]:
        """Get completed keys."""
        if not self._loaded:
            self.load()
        return self._completed.copy()

    @property
    def stats(self) -> dict:
        """Get current stats."""
        return self._stats.copy()

    def __enter__(self):
        """Context manager entry - load state."""
        self.load()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - save state."""
        if self._pending_count > 0:
            self.save()
        return False
