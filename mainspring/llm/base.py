"""Deprecated: LLM contracts now live in ``ouroboros.ports`` (V0.3 consolidation).

Re-exported here for backward compatibility with V0.2 imports.
"""

from ouroboros.ports import ChatChunk, LLMGateway, LLMResponse, StreamingLLMGateway, ToolCall, ToolCallDelta, Usage

__all__ = [
    "LLMGateway",
    "LLMResponse",
    "StreamingLLMGateway",
    "ChatChunk",
    "ToolCall",
    "ToolCallDelta",
    "Usage",
]
