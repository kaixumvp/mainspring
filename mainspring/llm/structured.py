"""Structured output: schema enforcement shared by LLM gateways (V3.1).

``parse_structured`` is the framework-side contract: gateways may hint the provider through
``response_format``, but the returned content is always validated here, so every gateway in
the chain behaves identically regardless of provider support.
"""

from __future__ import annotations

from pydantic import BaseModel, ValidationError

from mainspring.ports import StructuredOutputError


def parse_structured(content: str | None, schema: type[BaseModel]):
    if content is None:
        raise StructuredOutputError(
            f"structured output requested but response has no content ({schema.__name__})"
        )
    try:
        return schema.model_validate_json(content)
    except ValidationError as exc:
        raise StructuredOutputError(f"response failed validation against {schema.__name__}") from exc


def schema_kw(schema) -> dict:
    """`{"schema": schema}` when requested, `{}` otherwise — decorators forward the keyword
    only when present so inner gateways that predate structured output keep working."""
    return {"schema": schema} if schema is not None else {}


__all__ = ["parse_structured", "schema_kw"]
