"""Zero-dependency DevUI server: stdlib ``ThreadingHTTPServer`` + SSE + a single-page UI.

Serves the offline demo agent (``mainspring.devui.demo``) over three endpoints:

  * ``GET /``          — the single-file DevUI page (timeline, live deltas, approvals)
  * ``GET /events``    — SSE stream of ``RuntimeEvent``s (the runtime's EventSink bridges here)
  * ``POST /run``      — start the demo agent; returns the ``RunResult`` (pauses on approval)
  * ``POST /approve``  — resume a paused run with the granted ``resource_id``s

An asyncio loop runs in a background thread (the only threading in play); requests are
marshalled onto it with ``run_coroutine_threadsafe``. Everything is stdlib.
"""

from __future__ import annotations

import asyncio
import json
import queue
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from mainspring.core.runtime import AgentRuntime, RunContext
from mainspring.devui.demo import DEMO_TASK, build_demo_definition, build_demo_deps

_STATIC_DIR = Path(__file__).parent / "static"

_SSE_CONTENT_TYPE = "text/event-stream; charset=utf-8"
_PING_SECONDS = 15


class DevUIServer:
    """Hosts the DevUI. ``start()`` is non-blocking; ``stop()`` tears everything down.

    ``stream_delay`` throttles the demo's word-by-word streaming (0 in tests). With
    ``port=0`` the OS picks a free port, exposed via ``self.port`` after ``start()``.
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 8765, *, stream_delay: float = 0.02) -> None:
        self.host = host
        self.port = port
        self.stream_delay = stream_delay
        self._checkpointer = None  # built lazily on the loop thread
        self._runtime = AgentRuntime()
        self._definition = build_demo_definition()
        self._deps_by_run: dict[str, object] = {}  # resume must reuse the run's scripted gateway
        self._deps_lock = threading.Lock()
        self._bus: queue.Queue = queue.Queue()
        self._subscribers: set[queue.Queue] = set()
        self._sub_lock = threading.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._threads: list[threading.Thread] = []
        self._httpd: ThreadingHTTPServer | None = None
        self._stopped = threading.Event()
        self._ready = threading.Event()

    # ------------------------------------------------------------------ lifecycle
    def start(self) -> None:
        if self._httpd is not None:
            return
        handler = self._make_handler()
        self._httpd = ThreadingHTTPServer((self.host, self.port), handler)
        self._httpd.app = self
        self._httpd.daemon_threads = True
        self.port = self._httpd.server_address[1]

        self._threads.append(threading.Thread(target=self._run_loop, name="devui-loop", daemon=True))
        self._threads.append(threading.Thread(target=self._dispatch, name="devui-bus", daemon=True))
        self._threads.append(threading.Thread(target=self._httpd.serve_forever, name="devui-http", daemon=True))
        for t in self._threads:
            t.start()
        self._ready.wait(timeout=5)

    def stop(self) -> None:
        self._stopped.set()
        self._bus.put(None)
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
            self._httpd = None
        for t in self._threads:
            t.join(timeout=5)

    @property
    def address(self) -> str:
        host = "127.0.0.1" if self.host in ("0.0.0.0", "::") else self.host
        return f"http://{host}:{self.port}"

    def url(self, path: str = "/") -> str:
        return f"{self.address}{path}"

    # ------------------------------------------------------------------ internals
    def _make_handler(self) -> type[BaseHTTPRequestHandler]:
        server = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, format, *args):  # quiet, unless the app enables logging
                pass

            def do_GET(self):  # noqa: N802 — http.server API
                if self.path.startswith("/events"):
                    return self._serve_sse()
                if self.path in ("/", "/index.html"):
                    return self._serve_static()
                self.send_error(404)

            def do_POST(self):  # noqa: N802 — http.server API
                body = self._read_body()
                try:
                    payload = json.loads(body) if body else {}
                except json.JSONDecodeError:
                    return self._send_json(400, {"error": "invalid JSON body"})
                if self.path == "/run":
                    return self._handle_run(payload)
                if self.path == "/approve":
                    return self._handle_approve(payload)
                self.send_error(404)

            # ---------------------------------------------------------- helpers
            def _read_body(self) -> bytes:
                length = int(self.headers.get("Content-Length") or 0)
                return self.rfile.read(length) if length else b""

            def _send_json(self, code: int, payload: dict) -> None:
                data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                self.send_response(code)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def _serve_static(self) -> None:
                try:
                    data = (_STATIC_DIR / "index.html").read_bytes()
                except OSError:
                    return self.send_error(500, "index.html missing")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def _serve_sse(self) -> None:
                # Subscribe before the headers go out: once the client's request returns,
                # it is guaranteed to be registered (no event race).
                q: queue.Queue = queue.Queue(maxsize=1024)
                with server._sub_lock:
                    server._subscribers.add(q)
                self.send_response(200)
                self.send_header("Content-Type", _SSE_CONTENT_TYPE)
                self.send_header("Cache-Control", "no-cache")
                self.send_header("X-Accel-Buffering", "no")
                self.end_headers()
                try:
                    while not server._stopped.is_set():
                        try:
                            event = q.get(timeout=_PING_SECONDS)
                        except queue.Empty:
                            self.wfile.write(b": ping\n\n")
                            self.wfile.flush()
                            continue
                        payload = json.dumps(event, ensure_ascii=False, default=str)
                        self.wfile.write(f"data: {payload}\n\n".encode("utf-8"))
                        self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError, OSError):
                    pass  # client went away
                finally:
                    with server._sub_lock:
                        server._subscribers.discard(q)

            def _handle_run(self, payload: dict) -> None:
                task = str(payload.get("task") or DEMO_TASK)
                if not server._loop:
                    return self._send_json(503, {"error": "server not ready"})
                run_id = f"run_{uuid.uuid4().hex}"
                deps = build_demo_deps(
                    server._sink, stream_delay=server.stream_delay, checkpointer=server._checkpointer
                )
                with server._deps_lock:
                    server._deps_by_run[run_id] = deps
                context = RunContext(tenant_id="devui", user_id="demo", session_id="devui", run_id=run_id)
                future = asyncio.run_coroutine_threadsafe(
                    server._runtime.run(server._definition, task, context, deps), server._loop
                )
                try:
                    result = future.result(timeout=180)
                except Exception as exc:  # noqa: BLE001 — report as 500, keep server alive
                    return self._send_json(500, {"error": str(exc)})
                if result.status != "awaiting_human":
                    with server._deps_lock:
                        server._deps_by_run.pop(run_id, None)
                self._send_json(200, result.model_dump())

            def _handle_approve(self, payload: dict) -> None:
                run_id = payload.get("run_id")
                approvals = payload.get("approvals") or []
                if not run_id or not server._loop:
                    return self._send_json(400, {"error": "run_id is required"})
                with server._deps_lock:
                    deps = server._deps_by_run.get(run_id)
                if deps is None:
                    return self._send_json(404, {"error": f"unknown run_id: {run_id}"})
                future = asyncio.run_coroutine_threadsafe(
                    server._runtime.resume(server._definition, run_id, approvals, deps),
                    server._loop,
                )
                try:
                    result = future.result(timeout=180)
                except Exception as exc:  # noqa: BLE001
                    return self._send_json(500, {"error": str(exc)})
                if result.status != "awaiting_human":
                    with server._deps_lock:
                        server._deps_by_run.pop(run_id, None)
                self._send_json(200, result.model_dump())

        return Handler

    # ------------------------------------------------------------------ event plumbing
    async def _sink(self, event) -> None:
        """EventSink for the runtime: push JSON-ready event dicts onto the bus."""
        self._bus.put_nowait(event.model_dump())

    def _dispatch(self) -> None:
        """Fan bus events out to every SSE subscriber (slow subscribers drop events)."""
        while True:
            event = self._bus.get()
            if event is None:
                break
            with self._sub_lock:
                subscribers = list(self._subscribers)
            for q in subscribers:
                try:
                    q.put_nowait(event)
                except queue.Full:
                    pass

    def _run_loop(self) -> None:
        from mainspring.core.checkpointer import InMemoryCheckpointer

        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._checkpointer = InMemoryCheckpointer()
        self._ready.set()
        self._loop.run_forever()


__all__ = ["DevUIServer"]
