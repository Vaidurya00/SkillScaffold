"""Stop only ScienceWorld services listed in one run-owned PID file."""

from __future__ import annotations

import argparse
import json
import os
import signal
import time


def _cmdline(pid: int) -> str:
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as handle:
            return handle.read().replace(b"\0", b" ").decode(errors="replace")
    except OSError:
        return ""


def _stop(pid: int) -> None:
    command = _cmdline(pid)
    if not command or "env/scienceworld/" not in command:
        return
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.time() + 5
    while time.time() < deadline:
        try:
            os.kill(pid, 0)
        except OSError:
            return
        time.sleep(0.1)
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pid-file", required=True)
    args = parser.parse_args()
    if not os.path.isfile(args.pid_file):
        return
    payload = json.loads(open(args.pid_file).read())
    entries = [payload.get("router"), *payload.get("workers", [])]
    for entry in entries:
        if isinstance(entry, dict) and isinstance(entry.get("pid"), int):
            _stop(entry["pid"])
    try:
        os.unlink(args.pid_file)
    except FileNotFoundError:
        pass


if __name__ == "__main__":
    main()
