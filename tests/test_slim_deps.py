"""Deterministic tests for v3.6 optional-dependency slimming (no LLM, no network).

Each test runs a child Python process whose meta path raises ImportError for the
blocked heavy dependency, simulating a host that never installed the extras.
"""

import subprocess
import sys
import textwrap

BLOCKED = ["litellm", "redis", "httpx"]

_BLOCKER = textwrap.dedent(
    """
    import importlib.machinery
    import sys

    class _BlockingLoader:
        def create_module(self, spec):
            raise ImportError("blocked optional dependency: " + spec.name)

        def exec_module(self, module):
            raise ImportError("blocked optional dependency: " + spec.name)

    class _Block:
        def find_spec(self, name, path=None, target=None):
            if name.split(".")[0] in {blocked!r}:
                return importlib.machinery.ModuleSpec(name, _BlockingLoader())
            return None

    sys.meta_path.insert(0, _Block())
    """
).format(blocked=BLOCKED)


def _run(code: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", _BLOCKER + textwrap.dedent(code)],
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_import_mainspring_without_heavy_deps():
    proc = _run(
        """
        import mainspring
        print("ok")
        """
    )
    assert proc.returncode == 0, proc.stderr
    assert "ok" in proc.stdout


def test_litellm_gateway_imports_but_use_raises_actionable_error():
    proc = _run(
        """
        import asyncio
        from mainspring.llm import LiteLLMGateway

        async def main():
            try:
                await LiteLLMGateway(model="m").chat(messages=[])
            except ImportError as exc:
                print(str(exc))
                return
            raise SystemExit("expected ImportError")

        asyncio.run(main())
        """
    )
    assert proc.returncode == 0, proc.stderr
    assert "mainspring[litellm]" in proc.stdout


def test_embedder_use_raises_actionable_error():
    proc = _run(
        """
        import asyncio
        from mainspring.llm.embedder import LiteLLMEmbedder

        async def main():
            try:
                await LiteLLMEmbedder(model="m").embed(["x"])
            except ImportError as exc:
                print(str(exc))
                return
            raise SystemExit("expected ImportError")

        asyncio.run(main())
        """
    )
    assert proc.returncode == 0, proc.stderr
    assert "mainspring[litellm]" in proc.stdout


def test_http_tools_raise_actionable_error():
    proc = _run(
        """
        import asyncio
        from mainspring.tools._http import http_request
        from mainspring.tools.mcp import McpClient

        async def main():
            try:
                await http_request("GET", "http://example.invalid", timeout=1)
            except ImportError as exc:
                print("http_request:", exc)
            try:
                await McpClient("http://example.invalid").initialize()
            except ImportError as exc:
                print("mcp:", exc)
                return
            raise SystemExit("expected ImportError")

        asyncio.run(main())
        """
    )
    assert proc.returncode == 0, proc.stderr
    assert "mainspring[http]" in proc.stdout


def test_working_memory_imports_without_redis():
    proc = _run(
        """
        from mainspring.memory import WorkingBrain
        from mainspring.memory.working import WorkingBrain as WB2

        class FakeRedis:
            async def get(self, key):
                return None

            async def set(self, key, value, ex=None):
                pass

        brain = WorkingBrain(FakeRedis())
        print("ok")
        """
    )
    assert proc.returncode == 0, proc.stderr
    assert "ok" in proc.stdout
