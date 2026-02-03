"""
Base handler class for upload handlers.
"""

from __future__ import annotations

import logging
from typing import Dict, Any


class BaseHandler:
    """Base class for upload handlers with common utilities."""

    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.stats = {'success': 0, 'failed': 0, 'skipped': 0, 'canceled': 0}

    def reset_stats(self):
        """Reset statistics counters."""
        self.stats = {'success': 0, 'failed': 0, 'skipped': 0, 'canceled': 0}

    def log_summary(self, task_name: str, total: int, elapsed: float) -> Dict[str, Any]:
        """Log summary statistics."""
        rate = total / elapsed if elapsed > 0 else 0
        self.logger.info(
            f"{task_name} completed in {elapsed:.1f}s: "
            f"{self.stats['success']} success, {self.stats['failed']} failed, "
            f"{self.stats['skipped']} skipped, {self.stats['canceled']} canceled "
            f"({rate:.2f} items/sec)"
        )
        return {**self.stats, 'total': total, 'elapsed': elapsed, 'rate': rate}
