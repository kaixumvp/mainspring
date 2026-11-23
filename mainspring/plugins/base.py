"""Plugin protocol and adapter.

The protocol + manifest + invoke context are the framework↔host contract and now live in
``mainspring.ports`` (V0.3). This module keeps the concrete ``ToolAsPlugin`` adapter and
re-exports the contract names for backward compatibility.
"""

from typing import Any

from mainspring.ports import InvokeContext, Plugin, PluginManifest

__all__ = ["InvokeContext", "Plugin", "PluginManifest", "ToolAsPlugin"]


def _default_json_schema() -> dict:
    return {"type": "object", "properties": {}, "required": []}


class ToolAsPlugin:
    """Adapts a legacy V0.1 Tool (run-based) to the Plugin protocol (invoke-based).

    Tools keep their own `run()` API; this adapter is the bridge that lets the plugin
    registry treat tools and skills uniformly.
    """

    kind = "tool"

    def __init__(self, tool: Any) -> None:
        self._tool = tool
        self.manifest: PluginManifest = PluginManifest(
            name=tool.name,
            resource_id=(
                getattr(tool, "resource_id", None)
                or getattr(tool, "id", None)
                or tool.name
            ),
            version="1",
            kind="tool",
            description=getattr(tool, "description", ""),
            parameters=getattr(tool, "parameters", _default_json_schema()),
            permission=getattr(tool, "permission", "read"),
        )

    async def invoke(self, ctx: InvokeContext, **kw: Any) -> Any:
        return await self._tool.run(**kw)
