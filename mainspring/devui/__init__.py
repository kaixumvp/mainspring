"""DevUI: zero-dependency developer console for mainspring agents.

``mainspring dev`` serves a single-page UI that visualizes a run's steps,
token-level ``message.delta`` streams, and the human-in-the-loop approval flow,
driven by a fully offline scripted demo agent (no API key required).
"""

from mainspring.devui.demo import build_demo_definition, build_demo_deps
from mainspring.devui.server import DevUIServer

__all__ = ["DevUIServer", "build_demo_definition", "build_demo_deps"]
