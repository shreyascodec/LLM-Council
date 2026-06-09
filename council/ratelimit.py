"""Process-wide rate limiter for the free tier (~20 req/min, ~200 req/day).

A single limiter fronts every call so the generator fan-out can't exceed the
per-minute ceiling, and the run hard-stops before exhausting the daily budget.
"""
from __future__ import annotations

import threading
import time
from collections import deque


class DailyBudgetExceeded(RuntimeError):
    """Raised when the configured daily request budget would be exceeded."""


class RateLimiter:
    """Sliding-window limiter: at most `rpm` calls in any 60s window, and at
    most `rpd` calls per process run. Thread-safe so the generator fan-out can
    share one budget."""

    def __init__(self, requests_per_minute: int, requests_per_day: int):
        self._rpm = max(1, int(requests_per_minute))
        self._rpd = max(1, int(requests_per_day))
        self._lock = threading.Lock()
        self._minute_window: deque[float] = deque()
        self._day_count = 0

    @property
    def requests_used_today(self) -> int:
        return self._day_count

    def acquire(self) -> None:
        """Block until a request slot is available; reserve it before returning."""
        while True:
            with self._lock:
                now = time.monotonic()
                # drop timestamps older than 60s
                while self._minute_window and now - self._minute_window[0] >= 60.0:
                    self._minute_window.popleft()

                if self._day_count >= self._rpd:
                    raise DailyBudgetExceeded(
                        f"Daily request budget of {self._rpd} reached. "
                        f"Free tier is ~200/day; resume tomorrow or lower call count."
                    )

                if len(self._minute_window) < self._rpm:
                    self._minute_window.append(now)
                    self._day_count += 1
                    return

                # need to wait until the oldest call exits the 60s window
                sleep_for = 60.0 - (now - self._minute_window[0]) + 0.01
            time.sleep(max(0.05, sleep_for))
