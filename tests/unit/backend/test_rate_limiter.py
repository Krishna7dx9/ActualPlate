import asyncio

from backend.app.services.rate_limiter import RateLimiter


def test_in_memory_rate_limiter_enforces_window_and_isolates_identities():
    async def exercise():
        limiter = RateLimiter(redis_url=None, limit=2, window_seconds=60)
        first = await limiter.check("api-key:one")
        second = await limiter.check("api-key:one")
        third = await limiter.check("api-key:one")
        other = await limiter.check("api-key:two")
        await limiter.close()
        return first, second, third, other

    first, second, third, other = asyncio.run(exercise())
    assert first == (True, 1, 0)
    assert second == (True, 0, 0)
    assert third[0] is False
    assert third[1] == 0
    assert third[2] >= 1
    assert other == (True, 1, 0)
