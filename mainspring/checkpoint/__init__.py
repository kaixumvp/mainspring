"""Official Checkpointer implementations (V3.3): SQLite, Redis, PostgreSQL.

All three persist the same ``{"schema_version": N, "state": ...}`` envelope (`codec.py`)
and satisfy the ``mainspring.ports.Checkpointer`` protocol:

- ``SQLiteCheckpointer`` — zero-configuration, zero-dependency, single file. The default
  for standalone hosts that need durability without any infrastructure.
- ``RedisCheckpointer`` — cross-process / cross-instance over a host-supplied
  `redis.asyncio.Redis` client (no new dependency; the Working Brain uses the same pattern).
- ``PostgresCheckpointer`` — durable snapshots over a host-supplied DBAPI 2.0 connection;
  the framework never imports a PG driver. Hosts that need the snapshot colocated with
  their own run row (atomic status + snapshot) should implement the port themselves.

``InMemoryCheckpointer`` remains the test default and lives in
``mainspring.core.checkpointer``.
"""

from mainspring.checkpoint.codec import SCHEMA_VERSION, decode, encode
from mainspring.checkpoint.postgres import PostgresCheckpointer
from mainspring.checkpoint.redis import RedisCheckpointer
from mainspring.checkpoint.sqlite import SQLiteCheckpointer

__all__ = [
    "SCHEMA_VERSION",
    "PostgresCheckpointer",
    "RedisCheckpointer",
    "SQLiteCheckpointer",
    "decode",
    "encode",
]
