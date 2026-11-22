"""PostgreSQL Checkpointer (V3.3): durable snapshots for hosts that already run Postgres.

The framework stays dependency-free: it accepts any DBAPI 2.0 connection (``psycopg2``,
``psycopg``, …) supplied by the host, so importing ``mainspring`` never requires a PG
driver. All DB work runs on a worker thread; the table is created lazily on first use.

This keeps its own ``checkpoints`` table and is *not* a substitute for colocating the
snapshot on the host's run row with an atomic status flip — hosts that need
"status + snapshot in one transaction" should implement the port themselves (the
business layer's `DbCheckpointer` does exactly that).
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

from mainspring.checkpoint.codec import SCHEMA_VERSION, decode, encode

_TABLE_DDL = """
CREATE TABLE IF NOT EXISTS checkpoints (
    run_id TEXT PRIMARY KEY,
    state TEXT NOT NULL,
    schema_version INTEGER NOT NULL,
    updated_at TEXT NOT NULL
)
"""

_UPSERT_SQL = """
INSERT INTO checkpoints (run_id, state, schema_version, updated_at)
VALUES (%s, %s, %s, %s)
ON CONFLICT (run_id) DO UPDATE SET
    state = EXCLUDED.state,
    schema_version = EXCLUDED.schema_version,
    updated_at = EXCLUDED.updated_at
"""


class PostgresCheckpointer:
    """Persists run state snapshots over a host-supplied DBAPI 2.0 connection."""

    def __init__(self, conn) -> None:
        self._conn = conn
        self._ready = False
        self._lock = asyncio.Lock()

    def _execute(self, sql: str, params: tuple = ()):
        cur = self._conn.cursor()
        try:
            cur.execute(sql, params)
            self._conn.commit()
            return cur
        except Exception:
            self._conn.rollback()
            raise

    async def _ensure_table(self) -> None:
        async with self._lock:
            if self._ready:
                return
            await asyncio.to_thread(self._execute, _TABLE_DDL)
            self._ready = True

    async def save(self, run_id: str, state: dict) -> None:
        await self._ensure_table()
        now = datetime.now(timezone.utc).isoformat()
        await asyncio.to_thread(self._execute, _UPSERT_SQL, (run_id, encode(state), SCHEMA_VERSION, now))

    async def load(self, run_id: str) -> dict | None:
        await self._ensure_table()
        cur = await asyncio.to_thread(
            self._execute, "SELECT state FROM checkpoints WHERE run_id = %s", (run_id,)
        )
        row = cur.fetchone()
        return None if row is None else decode(row[0])

    async def delete(self, run_id: str) -> None:
        await self._ensure_table()
        await asyncio.to_thread(self._execute, "DELETE FROM checkpoints WHERE run_id = %s", (run_id,))

    async def list_runs(self) -> list[str]:
        await self._ensure_table()
        cur = await asyncio.to_thread(
            self._execute, "SELECT run_id FROM checkpoints ORDER BY run_id"
        )
        return [row[0] for row in cur.fetchall()]


__all__ = ["PostgresCheckpointer"]
