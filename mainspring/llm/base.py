"""Deprecated: LLM contracts now live in ``mainspring.ports`` (V0.3 consolidation).

Re-exported here for backward compatibility with V0.2 imports.
"""

from mainspring.ports import (
    ChatChunk,
    LLMGateway,
    LLMResponse,
    StreamingLLMGateway,
    StructuredOutputError,
    ToolCall,
    ToolCallDelta,
    Usage,
)

__all__ = [
    "LLMGateway",
    "LLMResponse",
    "StreamingLLMGateway",
    "ChatChunk",
    "ToolCall",
    "ToolCallDelta",
    "Usage",
    "StructuredOutputError",
]
