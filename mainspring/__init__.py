"""Ouroboros — a business-agnostic agent runtime."""

import logging

__version__ = "2.1.0"

from ouroboros.context import SlidingWindowContext  # noqa: E402
from ouroboros.core.runtime import (  # noqa: E402
    AgentDefinition,
    AgentRuntime,
    RunContext,
    RunResult,
    RuntimeConfig,
    RuntimeDeps,
    Step,
)

__all__ = [
    "__version__",
    "AgentRuntime",
    "AgentDefinition",
    "RunContext",
    "RunResult",
    "RuntimeConfig",
    "RuntimeDeps",
    "SlidingWindowContext",
    "Step",
]

# Library best practice: attach a NullHandler so an unconfigured host doesn't emit
# "No handlers could be found" warnings from our loggers.
logging.getLogger("ouroboros").addHandler(logging.NullHandler())
