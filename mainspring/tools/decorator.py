"""``@tool`` decorator: a typed function becomes a ``BaseTool`` (V3.2).

Signature + docstring become the tool contract: the parameter JSON Schema is Pydantic's
callable schema (types, defaults, `Annotated[T, Field(description=...)]` descriptions,
`Literal` → enum), the docstring becomes the tool description, and every call is validated
and coerced before execution. Sync functions run on a worker thread so the event loop
never blocks; async functions run inline.
"""

from __future__ import annotations

import asyncio
import inspect
from typing import Any, Callable

from pydantic import TypeAdapter, ValidationError, validate_call

from mainspring.tools.base import BaseTool, ToolResult


def _function_schema(fn) -> dict:
    """Pydantic's callable schema is already an OpenAI-style parameters object."""
    schema = TypeAdapter(fn).json_schema()
    schema.pop("title", None)
    return schema


class FunctionTool(BaseTool):
    """A ``BaseTool`` built from a plain function by ``@tool``."""

    def __init__(
        self,
        fn: Callable,
        *,
        name: str | None = None,
        description: str | None = None,
        permission: str = "read",
    ) -> None:
        self.fn = fn
        self.id = f"tool_{name or fn.__name__}"
        self.name = name or fn.__name__
        self.description = description if description is not None else inspect.getdoc(fn) or ""
        self.permission = permission
        self.parameters = _function_schema(fn)
        self._wrapped = validate_call(fn)
        self._async = inspect.iscoroutinefunction(fn)

    async def run(self, **kwargs: Any) -> ToolResult:
        try:
            if self._async:
                result = await self._wrapped(**kwargs)
            else:
                result = await asyncio.to_thread(self._wrapped, **kwargs)
        except ValidationError as exc:
            return ToolResult(success=False, output="", error=str(exc))
        except Exception as exc:  # noqa: BLE001 — tool contract: failures become ToolResult
            return ToolResult(success=False, output="", error=str(exc))
        if isinstance(result, ToolResult):
            return result
        return ToolResult(
            success=True,
            output=result if isinstance(result, (str, dict)) else str(result),
        )


def tool(fn=None, *, name: str | None = None, description: str | None = None,
         permission: str = "read") -> FunctionTool | Callable[[Callable], FunctionTool]:
    """Turn a typed function into a ``FunctionTool``.

    Usable bare (``@tool``) or parameterized (``@tool(name="...", description="...",
    permission="write")``). The decorated function is still reachable as ``tool.fn``.
    """
    if fn is None:
        return lambda f: tool(f, name=name, description=description, permission=permission)
    return FunctionTool(fn, name=name, description=description, permission=permission)


__all__ = ["FunctionTool", "tool"]
