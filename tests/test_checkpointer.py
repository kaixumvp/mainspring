"""Checkpointer port: one shared semantic suite across all four backends.

The four backends (in-memory, SQLite, Redis, PostgreSQL) must be behaviorally
interchangeable — a host swapping one for another must not change resume semantics — so
the core contract tests run against every backend through the same fixture. The envelope
tests (schema version / corrupt payload) run only against the durable backends, which
actually serialize; per-backend quirks (durability across reopen, TTL, upsert) have their
own tests. Redis and PostgreSQL are faked, keeping the suite dependency-free.
"""

import sqlite3

import pytest

from mainspring.checkpoint import PostgresCheckpointer, RedisCheckpointer, SQLiteCheckpointer
from mainspring.core.checkpointer import InMemoryCheckpointer

STATE = {"run_id": "r1", "messages": [{"role": "user", "content": "hi"}], "n": 1}


class FakeRedis:
    def __init__(self):
        self.store: dict = {}
        self.ttls: dict = {}

    async def get(self, key):
        return self.store.get(key)

    async def set(self, key, value, ex=None):
        self.store[key] = value
        if ex is not None:
            self.ttls[key] = ex

    async def delete(self, key):
        self.store.pop(key, None)

    async def scan_iter(self, match="*"):
        prefix = match.rstrip("*")
        for key in list(self.store):
            if key.startswith(prefix):
                yield key


class FakePostgresCursor:
    def __init__(self, conn):
        self._cur = conn._sqlite.cursor()

    def execute(self, sql, params=()):
        self._cur.execute(sql.replace("%s", "?"), params)
        return self

    def fetchone(self):
        return self._cur.fetchone()

    def fetchall(self):
        return self._cur.fetchall()


class FakePostgresConnection:
    """DBAPI 2.0 double: runs the real PG-dialect SQL against in-memory SQLite
    (``%s`` → ``?``), so upsert/select/delete semantics are exercised for real."""

    def __init__(self):
        # Real PG drivers are thread-safe across to_thread calls; SQLite is not, so the
        # double opts out of its thread check to mirror that behavior.
        self._sqlite = sqlite3.connect(":memory:", check_same_thread=False)

    def cursor(self):
        return FakePostgresCursor(self)

    def commit(self):
        self._sqlite.commit()

    def rollback(self):
        self._sqlite.rollback()


@pytest.fixture(params=["memory", "sqlite", "redis", "postgres"])
def checkpointer(request, tmp_path):
    if request.param == "memory":
        return InMemoryCheckpointer()
    if request.param == "sqlite":
        return SQLiteCheckpointer(tmp_path / "cp.db")
    if request.param == "redis":
        return RedisCheckpointer(FakeRedis())
    return PostgresCheckpointer(FakePostgresConnection())


@pytest.fixture(params=["sqlite", "redis", "postgres"])
def durable(request, tmp_path):
    """(checkpointer, write_raw) — write_raw injects a raw payload past the codec."""
    if request.param == "sqlite":
        path = tmp_path / "cp.db"
        cp = SQLiteCheckpointer(path)

        def write_raw(run_id, raw):
            conn = sqlite3.connect(path)
            conn.execute(
                "CREATE TABLE IF NOT EXISTS checkpoints (run_id TEXT PRIMARY KEY, state TEXT "
                "NOT NULL, schema_version INTEGER NOT NULL, updated_at TEXT NOT NULL)"
            )
            conn.execute(
                "INSERT INTO checkpoints (run_id, state, schema_version, updated_at) "
                "VALUES (?, ?, 1, '')",
                (run_id, raw),
            )
            conn.commit()
            conn.close()

        return cp, write_raw
    if request.param == "redis":
        fake = FakeRedis()
        cp = RedisCheckpointer(fake)
        return cp, lambda run_id, raw: fake.store.__setitem__(f"cp:{run_id}", raw)
    conn = FakePostgresConnection()
    # Pre-create the table the way a real host migration would; the checkpointer's own
    # lazy CREATE TABLE IF NOT EXISTS must coexist with it.
    cur = conn.cursor()
    cur.execute(
        "CREATE TABLE checkpoints (run_id TEXT PRIMARY KEY, state TEXT NOT NULL, "
        "schema_version INTEGER NOT NULL, updated_at TEXT NOT NULL)"
    )
    conn.commit()
    cp = PostgresCheckpointer(conn)

    def write_raw(run_id, raw):
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO checkpoints (run_id, state, schema_version, updated_at) "
            "VALUES (%s, %s, 1, '')",
            (run_id, raw),
        )
        conn.commit()

    return cp, write_raw


# --------------------------------------------------------------------------- shared semantics
@pytest.mark.asyncio
async def test_save_and_load_roundtrip(checkpointer):
    await checkpointer.save("r1", STATE)
    loaded = await checkpointer.load("r1")
    assert loaded == STATE
    assert loaded is not STATE  # snapshot, not the same object


@pytest.mark.asyncio
async def test_load_missing_returns_none(checkpointer):
    assert await checkpointer.load("nope") is None


@pytest.mark.asyncio
async def test_save_overwrites_previous_snapshot(checkpointer):
    await checkpointer.save("r1", {"n": 1})
    await checkpointer.save("r1", {"n": 2})
    assert (await checkpointer.load("r1"))["n"] == 2


@pytest.mark.asyncio
async def test_load_is_isolated_from_later_mutation(checkpointer):
    await checkpointer.save("r1", {"messages": [{"role": "user", "content": "a"}]})

    loaded = await checkpointer.load("r1")
    loaded["messages"][0]["content"] = "mutated"

    again = await checkpointer.load("r1")
    assert again["messages"][0]["content"] == "a"  # stored snapshot not corrupted


@pytest.mark.asyncio
async def test_delete_and_list_runs(checkpointer):
    await checkpointer.save("r1", {})
    await checkpointer.save("r2", {})
    assert set(await checkpointer.list_runs()) == {"r1", "r2"}

    await checkpointer.delete("r1")
    assert await checkpointer.load("r1") is None
    assert set(await checkpointer.list_runs()) == {"r2"}


@pytest.mark.asyncio
async def test_state_with_non_string_keys_and_values_roundtrips(checkpointer):
    await checkpointer.save("r1", {"steps": [{"i": 1, "ok": True}], "cost": 0.5})
    assert await checkpointer.load("r1") == {"steps": [{"i": 1, "ok": True}], "cost": 0.5}


# --------------------------------------------------------------------------- envelope (durable)
@pytest.mark.asyncio
async def test_unsupported_schema_version_raises(durable):
    cp, write_raw = durable
    write_raw("r1", '{"schema_version": 999, "state": {}}')

    with pytest.raises(ValueError, match="schema_version"):
        await cp.load("r1")


@pytest.mark.asyncio
async def test_corrupt_payload_raises(durable):
    cp, write_raw = durable
    write_raw("r1", "not-json")

    with pytest.raises(ValueError, match="not valid JSON"):
        await cp.load("r1")


# --------------------------------------------------------------------------- sqlite specifics
@pytest.mark.asyncio
async def test_sqlite_state_survives_reopen(tmp_path):
    path = tmp_path / "cp.db"
    await SQLiteCheckpointer(path).save("r1", STATE)

    # A brand-new instance (as after a process restart) sees the same snapshot.
    assert await SQLiteCheckpointer(path).load("r1") == STATE


def test_sqlite_rejects_memory_path():
    with pytest.raises(ValueError, match=":memory:"):
        SQLiteCheckpointer(":memory:")


# --------------------------------------------------------------------------- redis specifics
@pytest.mark.asyncio
async def test_redis_key_naming_and_default_no_ttl():
    fake = FakeRedis()
    await RedisCheckpointer(fake).save("r1", STATE)

    assert "cp:r1" in fake.store
    assert "cp:r1" not in fake.ttls


@pytest.mark.asyncio
async def test_redis_ttl_is_opt_in():
    fake = FakeRedis()
    await RedisCheckpointer(fake, ttl_seconds=60).save("r1", STATE)

    assert fake.ttls["cp:r1"] == 60
