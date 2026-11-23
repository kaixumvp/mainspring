"""Shared checkpoint serialization (V3.3): the one codec every backend uses.

State dicts are wrapped in a ``{"schema_version": N, "state": {...}}`` envelope so a
checkpoint written by an older framework version is *detected*, not silently resumed
with missing or misinterpreted fields. ``decode`` raises ``ValueError`` on corrupt JSON
or an unsupported version — resuming a half-approved run from a wrong snapshot is worse
than failing loudly.
"""

from __future__ import annotations

import json

SCHEMA_VERSION = 1


def encode(state: dict) -> str:
    return json.dumps(
        {"schema_version": SCHEMA_VERSION, "state": state},
        ensure_ascii=False,
        default=str,
    )


def decode(raw: str | bytes) -> dict:
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    try:
        envelope = json.loads(raw)
    except (json.JSONDecodeError, TypeError) as exc:
        raise ValueError("checkpoint payload is not valid JSON") from exc
    if not isinstance(envelope, dict) or envelope.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"checkpoint schema_version {envelope.get('schema_version') if isinstance(envelope, dict) else '?'!r} "
            f"is not supported (current: {SCHEMA_VERSION})"
        )
    return envelope["state"]


__all__ = ["SCHEMA_VERSION", "decode", "encode"]
