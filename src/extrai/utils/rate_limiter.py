import asyncio
import time
from typing import List, Tuple


class AsyncRateLimiter:
    """
    A generic async rate limiter using a sliding window algorithm.
    Tracks usage of a resource (calls, tokens, etc.) over a time period.
    """

    def __init__(self, max_capacity: int, period: float = 60.0):
        """
        Args:
            max_capacity: The maximum amount of resource allowed in the period.
            period: The time window in seconds (default 60.0 for 1 minute).
        """
        self.max_capacity = max_capacity
        self.period = period
        # List of (timestamp, cost)
        self.history: List[Tuple[float, int]] = []
        self._lock = asyncio.Lock()

    async def acquire(self, cost: int = 1):
        """
        Acquires the specified amount of resource, waiting if necessary.

        Args:
            cost: The amount of resource to consume (default 1).
        """
        async with self._lock:
            now = time.monotonic()

            # 1. Clean up old history
            self.history = [(t, c) for t, c in self.history if now - t <= self.period]

            # 2. Calculate current usage
            current_usage = sum(c for t, c in self.history)

            # 3. Check if we need to wait
            if current_usage + cost > self.max_capacity:
                # We need to wait until enough usage expires.
                # Find how much we need to free.
                needed_to_free = (current_usage + cost) - self.max_capacity

                freed = 0
                wait_until = now

                for t, c in self.history:
                    freed += c
                    if freed >= needed_to_free:
                        # Found the point where enough capacity is freed
                        wait_until = t + self.period
                        break

                sleep_time = wait_until - now
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)

                    # After sleep, update state
                    now = time.monotonic()
                    self.history = [
                        (t, c) for t, c in self.history if now - t <= self.period
                    ]

            # 4. Record usage
            self.history.append((now, cost))

    async def __aenter__(self):
        await self.acquire(1)
        return self

    async def __aexit__(self, exc_type, exc, tb):
        pass
