"""离线运行 ``example/`` 下全部示例并汇总结果。

用法：``python example/run_all.py``（使用当前解释器；建议先 ``poetry install``）
每个示例是独立子进程，互不污染；全部无需真实 API key。
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SKIP = {"_shared.py", "run_all.py"}
TIMEOUT_SECONDS = 120


def main() -> int:
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join([str(HERE.parent), str(HERE)]),
        "PYTHONIOENCODING": "utf-8",
    }
    paths = sorted(p for p in HERE.glob("*.py") if p.name not in SKIP)
    failures: list[tuple[str, str]] = []

    for path in paths:
        print(f"▶ {path.name}", flush=True)
        try:
            proc = subprocess.run(
                [sys.executable, str(path)],
                cwd=HERE,
                env=env,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired:
            failures.append((path.name, f"timeout after {TIMEOUT_SECONDS}s"))
            print("  ✗ timeout", flush=True)
            continue

        if proc.returncode == 0:
            print("  ✓ ok", flush=True)
            continue

        lines = (proc.stderr or proc.stdout).strip().splitlines()
        detail = lines[-1] if lines else f"exit code {proc.returncode}"
        failures.append((path.name, detail))
        print(f"  ✗ failed: {detail}", flush=True)

    passed = len(paths) - len(failures)
    print(f"\n{passed}/{len(paths)} 个示例通过")
    for name, detail in failures:
        print(f"  ✗ {name}: {detail}")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
