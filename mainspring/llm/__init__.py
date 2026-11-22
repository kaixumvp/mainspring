from mainspring.ports import LLMGateway, LLMResponse, StructuredOutputError, ToolCall, Usage
from mainspring.llm.cost import (
    CachingGateway,
    TieredRouter,
    TokenBudget,
    TokenBudgetExceeded,
    message_length_classifier,
)
from mainspring.llm.replay import ReplayGateway, RecordingGateway, dump_recording, load_recording
from mainspring.llm.routing import FallbackLLMGateway, FaultTolerantGateway
from mainspring.llm.structured import parse_structured

__all__ = [
    "LLMGateway",
    "LLMResponse",
    "StructuredOutputError",
    "ToolCall",
    "Usage",
    "LiteLLMGateway",
    "FallbackLLMGateway",
    "FaultTolerantGateway",
    "RecordingGateway",
    "ReplayGateway",
    "dump_recording",
    "load_recording",
    "parse_structured",
    "CachingGateway",
    "TieredRouter",
    "TokenBudget",
    "TokenBudgetExceeded",
    "message_length_classifier",
]


def __getattr__(name: str):
    # LiteLLMGateway pulls the optional `litellm` extra; import it lazily so
    # `import mainspring.llm` works without that dependency installed.
    if name == "LiteLLMGateway":
        from mainspring.llm.litellm_gateway import LiteLLMGateway

        return LiteLLMGateway
    raise AttributeError(f"module 'mainspring.llm' has no attribute {name!r}")
