"""`@tool` decorator (V3.2): typed functions become BaseTools with auto-generated schemas."""

from typing import Annotated, Literal

import pytest
from pydantic import Field, ValidationError

from mainspring.tools import FunctionTool, ToolRegistry, ToolResult, tool
from mainspring.plugins.base import ToolAsPlugin


def test_schema_is_derived_from_signature():
    @tool
    def add(a: int, b: int = 10) -> int:
        """Add two numbers."""
        return a + b

    assert isinstance(add, FunctionTool)
    assert add.name == "add"
    assert add.description == "Add two numbers."
    assert add.parameters == {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "a": {"title": "A", "type": "integer"},
            "b": {"default": 10, "title": "B", "type": "integer"},
        },
        "required": ["a"],
    }


def test_field_description_and_literal_become_schema():
    @tool
    def search(q: Annotated[str, Field(description="query")], mode: Literal["fast", "deep"] = "fast") -> str:
        return f"{q} {mode}"

    props = search.parameters["properties"]
    assert props["q"]["description"] == "query"
    assert props["mode"]["enum"] == ["fast", "deep"]
    assert "mode" not in search.parameters["required"]


def test_overrides_name_description_permission():
    @tool(name="sum_up", description="custom", permission="write")
    def add(a: int, b: int) -> int:
        return a + b

    assert add.name == "sum_up"
    assert add.id == "tool_sum_up"
    assert add.description == "custom"
    assert add.permission == "write"
    assert add.fn(1, 2) == 3  # original function stays reachable


@pytest.mark.asyncio
async def test_sync_function_runs_and_coerces_types():
    @tool
    def add(a: int, b: int) -> int:
        return a + b

    result = await add.run(a="5", b=3)  # JSON args arrive as strings; validated + coerced

    assert result == ToolResult(success=True, output="8")


@pytest.mark.asyncio
async def test_async_function_runs_inline():
    @tool
    async def double(x: int) -> str:
        return f"{x * 2}"

    result = await double.run(x=4)

    assert result.output == "8"


@pytest.mark.asyncio
async def test_validation_failure_degrades_to_tool_result():
    @tool
    def add(a: int, b: int) -> int:
        return a + b

    result = await add.run(a="not-a-number", b=1)

    assert result.success is False
    assert result.error


@pytest.mark.asyncio
async def test_function_exception_degrades_to_tool_result():
    @tool
    def boom() -> str:
        raise RuntimeError("kaboom")

    result = await boom.run()

    assert result.success is False
    assert "kaboom" in result.error


@pytest.mark.asyncio
async def test_dict_and_toolresult_outputs_pass_through():
    @tool
    def as_dict() -> dict:
        return {"rows": 3}

    @tool
    def as_toolresult() -> ToolResult:
        return ToolResult(success=False, output="", error="no data")

    assert (await as_dict.run()).output == {"rows": 3}
    assert (await as_toolresult.run()).success is False


@pytest.mark.asyncio
async def test_registers_in_registry_and_adapts_to_plugin():
    @tool
    def add(a: int, b: int = 1) -> int:
        """Add."""
        return a + b

    registry = ToolRegistry()
    registry.register(add)

    assert registry.get("add") is add
    manifest = ToolAsPlugin(add).manifest

    assert manifest.name == "add"
    assert manifest.kind == "tool"
    assert manifest.description == "Add."
    assert manifest.parameters["required"] == ["a"]
