"""Shared Redis sliding-window limiter with a bounded local test/dev adapter."""
from __future__ import annotations

import asyncio
import hashlib
import math
import time
import uuid
from collections import deque

from redis.asyncio import Redis
from redis.exceptions import RedisError


_WINDOW_MS = 60_000
_MAX_LOCAL_IDENTITIES = 10_000
_REDIS_SCRIPT = """
local now = tonumber(ARGV[1])
local window = tonumber(ARGV[2])
local limit = tonumber(ARGV[3])
local member = ARGV[4]
redis.call('ZREMRANGEBYSCORE', KEYS[1], '-inf', now - window)
local count = redis.call('ZCARD', KEYS[1])
if count >= limit then
  local oldest = redis.call('ZRANGE', KEYS[1], 0, 0, 'WITHSCORES')
  local retry = window
  if oldest[2] then
    retry = math.max(1, tonumber(oldest[2]) + window - now)
  end
  return {0, count, retry}
end
redis.call('ZADD', KEYS[1], now, member)
redis.call('PEXPIRE', KEYS[1], window)
return {1, count + 1, 0}
"""


class RateLimiterUnavailable(RuntimeError):
    """Raised when the configured shared rate-limit store is unavailable."""


class RateLimiter:
    def __init__(self, *, redis_url: str | None, limit: int, window_seconds: int = 60):
        self.redis_url = redis_url
        self.limit = limit
        self.window_seconds = window_seconds
        self._redis: Redis | None = None
        self._local: dict[str, deque[float]] = {}
        self._lock = asyncio.Lock()

    async def check(self, identity: str) -> tuple[bool, int, int]:
        """Return (allowed, remaining, retry_after_seconds)."""
        identity_hash = hashlib.sha256(identity.encode("utf-8")).hexdigest()
        if self.redis_url:
            return await self._check_redis(identity_hash)
        return await self._check_local(identity_hash)

    async def _check_redis(self, identity_hash: str) -> tuple[bool, int, int]:
        if self._redis is None:
            try:
                self._redis = Redis.from_url(self.redis_url, decode_responses=True)
            except (RedisError, ValueError) as exc:
                raise RateLimiterUnavailable("Shared rate-limit storage is unavailable.") from exc
        now_ms = int(time.time() * 1000)
        redis_key = f"actualplate:rate-limit:v1:{identity_hash}"
        member = f"{now_ms}:{uuid.uuid4().hex}"
        try:
            result = await self._redis.eval(
                _REDIS_SCRIPT,
                1,
                redis_key,
                now_ms,
                self.window_seconds * 1000,
                self.limit,
                member,
            )
        except (RedisError, ValueError) as exc:
            raise RateLimiterUnavailable("Shared rate-limit storage is unavailable.") from exc
        allowed = bool(int(result[0]))
        count = int(result[1])
        retry = max(0, math.ceil(int(result[2]) / 1000))
        return allowed, max(0, self.limit - count), retry

    async def _check_local(self, identity_hash: str) -> tuple[bool, int, int]:
        now = time.monotonic()
        cutoff = now - self.window_seconds
        async with self._lock:
            events = self._local.get(identity_hash)
            if events is None:
                if len(self._local) >= _MAX_LOCAL_IDENTITIES:
                    stale = [
                        key for key, values in self._local.items()
                        if not values or values[-1] <= cutoff
                    ]
                    for key in stale:
                        self._local.pop(key, None)
                if len(self._local) >= _MAX_LOCAL_IDENTITIES:
                    return False, 0, self.window_seconds
                events = deque()
                self._local[identity_hash] = events

            while events and events[0] <= cutoff:
                events.popleft()
            if len(events) >= self.limit:
                retry = max(1, math.ceil(events[0] + self.window_seconds - now))
                return False, 0, retry

            events.append(now)
            return True, max(0, self.limit - len(events)), 0

    async def health(self) -> dict[str, str]:
        if not self.redis_url:
            return {"status": "local", "mode": "in_memory"}
        if self._redis is None:
            self._redis = Redis.from_url(self.redis_url, decode_responses=True)
        try:
            await self._redis.ping()
        except RedisError as exc:
            raise RateLimiterUnavailable("Shared rate-limit storage is unavailable.") from exc
        return {"status": "ok", "mode": "redis"}

    async def close(self) -> None:
        if self._redis is not None:
            await self._redis.aclose()
            self._redis = None
