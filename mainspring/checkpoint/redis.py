"""Redis Checkpointer (V3.3): cross-process, cross-instance snapshots.

Takes a `redis.asyncio.Redis` client (already a framework dependency — the Working Brain
uses the same pattern), so no new dependency is introduced. Keys are ``{prefix}{run_id}``
with the state codec envelope as the value.

``ttl_seconds`` defaults to ``None`` (persist forever): a paused run waiting for human
approval must stay resumable for as long as the approval takes, and an expiry that
silently drops it would turn a late approval into a lost run. Pass a TTL only when you
have a separate cleanup policy for terminal runs.
"""

from __future__ import annotations

from mainspring.checkpoint.codec import decode, encode


class RedisCheckpointer:
    """Persists run state snapshots in Redis."""

    def __init__(self, redis, *, prefix: str = "cp:", ttl_seconds: int | None = None) -> None:
        self._redis = redis
        self._prefix = prefix
        self._ttl = ttl_seconds

    def _key(self, run_id: str) -> str:
        return f"{self._prefix}{run_id}"

    async def save(self, run_id: str, state: dict) -> None:
        await self._redis.set(self._key(run_id), encode(state), ex=self._ttl)

    async def load(self, run_id: str) -> dict | None:
        raw = await self._redis.get(self._key(run_id))
        return None if raw is None else decode(raw)

    async def delete(self, run_id: str) -> None:
        await self._redis.delete(self._key(run_id))

    async def list_runs(self) -> list[str]:
        keys = [key async for key in self._redis.scan_iter(match=f"{self._prefix}*")]
        return sorted(key[len(self._prefix):] for key in keys)


__all__ = ["RedisCheckpointer"]
