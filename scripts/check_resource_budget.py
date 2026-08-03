"""Production-start and memory-budget harness for free-tier deployment safety."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_COMMAND = [
    sys.executable,
    "-c",
    "import os; os.environ.setdefault('ENVIRONMENT','production'); from fastapi.testclient import TestClient; import api; r=TestClient(api.app).get('/health'); assert r.status_code == 200 and r.json()['status'] == 'ok'; print('ready', flush=True)",
]
BAD_EXIT_HINTS = {127: "command_not_found", 137: "oom_killed"}


def rss_bytes(pid: int) -> int | None:
    status = Path(f"/proc/{pid}/status")
    if status.exists():
        for line in status.read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-rss-mib", type=int, default=400)
    parser.add_argument("--limit-mib", type=int, default=512)
    parser.add_argument("--timeout", type=float, default=30)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command or DEFAULT_COMMAND
    env = os.environ.copy()
    env.setdefault("ENVIRONMENT", "production")
    env.setdefault("SERVER_MODELS_ENABLED", "false")
    started = time.monotonic()
    peak = 0
    restarts = 0
    process = subprocess.Popen(
        command,
        cwd=ROOT / "backend",
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        while process.poll() is None:
            current = rss_bytes(process.pid)
            if current is not None:
                peak = max(peak, current)
            if time.monotonic() - started > args.timeout:
                process.terminate()
                try:
                    process.wait(5)
                except subprocess.TimeoutExpired:
                    process.kill()
                break
            time.sleep(0.05)
        stdout, stderr = process.communicate(timeout=5)
    except Exception:
        process.kill()
        stdout, stderr = process.communicate()
        raise
    elapsed = round(time.monotonic() - started, 3)
    exit_code = process.returncode
    peak_mib = round(peak / 1024 / 1024, 2) if peak else None
    failures: list[str] = []
    if exit_code != 0:
        failures.append(BAD_EXIT_HINTS.get(exit_code, f"exit_{exit_code}"))
    if peak_mib is not None and peak_mib > args.max_rss_mib:
        failures.append("peak_rss_over_budget")
    if restarts:
        failures.append("unexpected_restart")
    if elapsed >= args.timeout and exit_code is None:
        failures.append("timeout")
    result = {
        "status": "passed" if not failures else "failed",
        "exit_code": exit_code,
        "failures": failures,
        "peak_rss_mib": peak_mib,
        "max_rss_mib": args.max_rss_mib,
        "limit_mib": args.limit_mib,
        "duration_seconds": elapsed,
        "stdout_tail": stdout[-500:],
        "stderr_tail": stderr[-500:],
    }
    if args.json:
        print(json.dumps(result, sort_keys=True))
    elif failures:
        print("resource budget failed: " + ", ".join(failures), file=sys.stderr)
    else:
        print("resource budget checks passed")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
