"""``mainspring`` command-line entry point.

Currently a single subcommand: ``mainspring dev`` starts the zero-dependency DevUI
(see ``mainspring.devui``) — a local single-page console for watching an agent run
step by step, token by token, including the human-in-the-loop approval pause.
"""

from __future__ import annotations

import argparse
import time


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="mainspring", description="Mainspring agent framework CLI")
    sub = parser.add_subparsers(dest="command")

    dev = sub.add_parser(
        "dev",
        help="start the local DevUI console (offline demo agent, no API key required)",
        description="Serve the DevUI: a single-page console visualizing the demo agent's "
        "steps, token stream, and approval flow.",
    )
    dev.add_argument("--host", default="127.0.0.1", help="bind address (default: 127.0.0.1)")
    dev.add_argument("--port", type=int, default=8765, help="port (default: 8765, 0 = pick a free port)")
    dev.add_argument(
        "--delay", type=float, default=0.02,
        help="seconds between streamed token chunks (default: 0.02; 0 = instant)",
    )

    args = parser.parse_args(argv)
    if args.command != "dev":
        parser.print_help()
        return 1

    from mainspring.devui.server import DevUIServer

    server = DevUIServer(host=args.host, port=args.port, stream_delay=args.delay)
    try:
        server.start()
    except OSError as exc:
        print(f"mainspring dev: failed to bind {args.host}:{args.port} — {exc}")
        return 1
    print(f"DevUI running at {server.address}  (Ctrl+C to stop)")
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
