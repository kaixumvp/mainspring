"""SQLite Checkpointer (V3.3): durable, zero-configuration, zero-dependency.

One file, one table, WAL mode. Each operation opens its own connection (the checkpoint
path is low-frequency, so per-call connections are simpler and safer than pooling);
SQLite's file locking handles cross-process safety. ``:memory:`` is rejected explicitly —
with per-call connections it would silently open a fresh empty database every call, which
is exactly the "state lost on restart" trap this class exists to fix.
"""

from __future__ import annotations

import asyncio
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

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
VALUES (?, ?, ?, ?)
ON CONFLICT (run_id) DO UPDATE SET
    state = excluded.state,
    schema_version = excluded.schema_version,
    updated_at = excluded.updated_at
"""


class SQLiteCheckpointer:
    """Persists run state snapshots in a single SQLite file (stdlib `sqlite3`)."""

    def __init__(self, path: str | Path) -> None:
        self._path = str(path)
        if self._path == ":memory:":
            raise ValueError(
                ":memory: is not durable — each call opens its own connection, so it would "
                "start empty every time. Pass a file path instead."
            )
        self._lock = threading.Lock()

    def _run_sync(self, fn):
        with self._lock:
            conn = sqlite3.connect(self._path)
            try:
                conn.execute("PRAGMA busy_timeout=5000")
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute(_TABLE_DDL)
                return fn(conn)
            finally:
                conn.close()

    async def save(self, run_id: str, state: dict) -> None:
        payload = encode(state)
        now = datetime.now(timezone.utc).isoformat()

        def _save(conn) -> None:
            conn.execute(_UPSERT_SQL, (run_id, payload, SCHEMA_VERSION, now))
            conn.commit()

        await asyncio.to_thread(self._run_sync, _save)

    async def load(self, run_id: str) -> dict | None:
        def _load(conn):
            row = conn.execute("SELECT state FROM checkpoints WHERE run_id = ?", (run_id,)).fetchone()
            return None if row is None else decode(row[0])

        return await asyncio.to_thread(self._run_sync, _load)

    async def delete(self, run_id: str) -> None:
        def _delete(conn) -> None:
            conn.execute("DELETE FROM checkpoints WHERE run_id = ?", (run_id,))
            conn.commit()

        await asyncio.to_thread(self._run_sync, _delete)

    async def list_runs(self) -> list[str]:
        def _list(conn):
            return [row[0] for row in conn.execute("SELECT run_id FROM checkpoints ORDER BY run_id")]

        return await asyncio.to_thread(self._run_sync, _list)


__all__ = ["SQLiteCheckpointer"]
