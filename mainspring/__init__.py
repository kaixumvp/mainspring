"""Mainspring — a business-agnostic agent runtime."""

import logging

__version__ = "3.6.0"

from mainspring.context import SlidingWindowContext
from mainspring.core.context import ContextPolicy
from mainspring.core.runtime import (
    AgentDefinition,
    AgentRuntime,
    RunContext,
    RunResult,
    RuntimeConfig,
    RuntimeDeps,
    Step,
)

__all__ = [
    "AgentDefinition",
    "AgentRuntime",
    "ContextPolicy",
    "RunContext",
    "RunResult",
    "RuntimeConfig",
    "RuntimeDeps",
    "SlidingWindowContext",
    "Step",
    "__version__",
]

# Library best practice: attach a NullHandler so an unconfigured host doesn't emit
# "No handlers could be found" warnings from our loggers.
logging.getLogger("mainspring").addHandler(logging.NullHandler())
