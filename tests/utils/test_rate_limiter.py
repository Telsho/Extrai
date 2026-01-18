import pytest
import asyncio
import time
from extrai.utils.rate_limiter import AsyncRateLimiter


@pytest.mark.asyncio
async def test_rate_limiter_basic():
    """Test basic RPM limiting."""
    limiter = AsyncRateLimiter(max_capacity=2, period=0.5)
    start = time.monotonic()

    # First 2 should be immediate
    await limiter.acquire(1)
    await limiter.acquire(1)

    # 3rd should wait
    await limiter.acquire(1)
    duration = time.monotonic() - start

    assert duration >= 0.5


@pytest.mark.asyncio
async def test_rate_limiter_tokens():
    """Test token-based limiting."""
    limiter = AsyncRateLimiter(max_capacity=10, period=0.5)
    start = time.monotonic()

    # Consuming 5 tokens twice
    await limiter.acquire(5)
    await limiter.acquire(5)

    # Consuming 1 more token should wait
    await limiter.acquire(1)
    duration = time.monotonic() - start

    assert duration >= 0.5


@pytest.mark.asyncio
async def test_rate_limiter_partial_wait():
    """Test waiting only for necessary capacity to be freed."""
    limiter = AsyncRateLimiter(max_capacity=10, period=1.0)

    await limiter.acquire(5)  # t=0
    # Wait half the period
    await asyncio.sleep(0.5)
    await limiter.acquire(5)  # t=0.5

    # Now usage is 10.
    # We want 5 more.
    # The first 5 expire at t=1.0. The second 5 expire at t=1.5.
    # We are currently at t=0.5.
    # We should wait until t=1.0 (0.5s wait) to free the first 5.

    start = time.monotonic()
    await limiter.acquire(5)
    duration = time.monotonic() - start

    # Should be close to 0.5s
    # Use a range to be safe against system timing jitter
    assert 0.4 <= duration <= 0.7


@pytest.mark.asyncio
async def test_rate_limiter_context_manager():
    """Test context manager usage."""
    limiter = AsyncRateLimiter(max_capacity=1, period=0.2)
    start = time.monotonic()

    async with limiter:
        pass

    # Should wait here
    async with limiter:
        pass

    duration = time.monotonic() - start
    assert duration >= 0.2
