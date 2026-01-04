"""
Rate limiting functionality for API request control.

This module provides a thread-safe rate limiter using token bucket algorithm
to ensure API requests stay within allowed rate limits.
"""

import time
import threading


class RateLimiter:
    """
    Thread-safe rate limiter for controlling API request rates.
    Implements token bucket algorithm for smooth rate limiting.

    Example:
        >>> limiter = RateLimiter(max_rate=10.0)  # 10 requests per second
        >>> limiter.acquire()  # Blocks if necessary to maintain rate limit
        >>> make_api_request()
    """

    def __init__(self, max_rate: float):
        """
        Initialize rate limiter.

        :param max_rate: Maximum requests per second (e.g., 10.0 for SEC API)
        """
        self.max_rate = max_rate
        self.min_interval = 1.0 / max_rate
        self.last_request_time = 0
        self.lock = threading.Lock()

    def acquire(self):
        """
        Acquire permission to make a request.
        Blocks (sleeps) if necessary to maintain rate limit.
        Thread-safe across multiple workers.
        """
        with self.lock:
            current_time = time.time()
            time_since_last = current_time - self.last_request_time

            if time_since_last < self.min_interval:
                sleep_time = self.min_interval - time_since_last
                time.sleep(sleep_time)
                self.last_request_time = time.time()
            else:
                self.last_request_time = current_time
