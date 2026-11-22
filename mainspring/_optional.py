"""Lazy loading for optional third-party dependencies (v3.6).

Heavy integrations (litellm, httpx, redis) ship as extras, not hard dependencies, so
``import mainspring`` stays light. Integration modules call :func:`require_optional` at use
time; when the extra is missing it raises an ImportError telling the host exactly what to
install.
"""

from importlib import import_module
from typing import Any

__all__ = ["require_optional"]


def require_optional(module: str, extra: str) -> Any:
    """Import an optional dependency or raise an actionable error naming the extra."""
    try:
        return import_module(module)
    except ImportError as exc:
        raise ImportError(
            f"Module {module!r} is not installed. Mainspring ships heavy integrations as "
            f"optional extras; install the one you need, e.g.: pip install 'mainspring[{extra}]'"
        ) from exc
